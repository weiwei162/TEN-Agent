//
// Copyright © 2025 Agora
// This file is part of TEN Framework, an open source project.
// Licensed under the Apache License, Version 2.0, with certain conditions.
// Refer to the "LICENSE" file in the root directory for more information.
//
use std::fs::{File, Metadata};
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Platform‐specific metadata extensions
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
#[cfg(windows)]
use std::os::windows::fs::MetadataExt;

use anyhow::{anyhow, Result};
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio::sync::oneshot;

use crate::log::process_log_line;
use crate::log::{GraphResourcesLog, LogLineInfo};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60); // 1 minute timeout.
const DEFAULT_BUFFER_SIZE: usize = 4096; // Default read buffer size.
const DEFAULT_CHECK_INTERVAL: Duration = Duration::from_millis(100);

/// Stream of UTF-8 text file content changes.
pub struct LogFileContentStream {
    // Channel for receiving file content as UTF-8 text.
    content_rx: Receiver<Result<LogLineInfo>>,

    // Sender to signal stop request.
    stop_tx: Option<oneshot::Sender<()>>,
}

impl LogFileContentStream {
    /// Create a new FileContentStream.
    fn new(
        content_rx: Receiver<Result<LogLineInfo>>,
        stop_tx: oneshot::Sender<()>,
    ) -> Self {
        Self { content_rx, stop_tx: Some(stop_tx) }
    }

    /// Get the next chunk of text from the file.
    pub async fn next(&mut self) -> Option<Result<LogLineInfo>> {
        self.content_rx.recv().await
    }

    /// Stop the file watching process.
    pub fn stop(&mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
    }
}

impl Drop for LogFileContentStream {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Options for watching a file.
#[derive(Clone)]
pub struct LogFileWatchOptions {
    /// Timeout for waiting for new content after reaching EOF.
    pub timeout: Duration,

    /// Size of buffer for reading.
    pub buffer_size: usize,

    /// Interval to check for new content when at EOF.
    pub check_interval: Duration,
}

impl Default for LogFileWatchOptions {
    fn default() -> Self {
        Self {
            timeout: DEFAULT_TIMEOUT,
            buffer_size: DEFAULT_BUFFER_SIZE,
            check_interval: DEFAULT_CHECK_INTERVAL,
        }
    }
}

/// Compare two metadata to determine if they point to the same file (used to
/// detect rotation).
fn is_same_file(a: &Metadata, b: &Metadata) -> bool {
    #[cfg(unix)]
    {
        a.dev() == b.dev() && a.ino() == b.ino()
    }
    #[cfg(windows)]
    {
        a.volume_serial_number() == b.volume_serial_number()
            && a.file_index() == b.file_index()
    }
}

/// Watch a UTF-8 text file for changes and stream its content.
///
/// Returns a FileContentStream that can be used to read the content of the file
/// as it changes. The stream will end when either:
/// 1. The caller stops it by calling `stop()` or dropping the stream.
/// 2. No new content is available after reaching EOF and the timeout is
///    reached.
pub async fn watch_log_file<P: AsRef<Path>>(
    path: P,
    options: Option<LogFileWatchOptions>,
) -> Result<LogFileContentStream> {
    let path = path.as_ref().to_path_buf();

    // Ensure the file exists before we start watching it.
    if !path.exists() {
        return Err(anyhow!("File does not exist: {}", path.display()));
    }

    let options = options.unwrap_or_default();

    // Create channels.
    let (content_tx, content_rx) = mpsc::channel(32);
    let (stop_tx, stop_rx) = oneshot::channel();

    // Spawn a task to watch the file.
    tokio::spawn(async move {
        watch_log_file_task(path, content_tx, stop_rx, options).await;
    });

    Ok(LogFileContentStream::new(content_rx, stop_tx))
}

/// Actual file watch task running in the background.
async fn watch_log_file_task(
    path: PathBuf,
    content_tx: Sender<Result<LogLineInfo>>,
    mut stop_rx: oneshot::Receiver<()>,
    options: LogFileWatchOptions,
) {
    // Create a GraphResourcesLog instance that will be used throughout this
    // task's lifetime.
    let mut graph_resources_log = GraphResourcesLog {
        graph_id: String::new(),
        graph_name: String::new(),
        app_uri: None,
        extension_threads: std::collections::HashMap::new(),
    };

    // Open the file.
    let mut file = match File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            let _ = content_tx.send(Err(anyhow::anyhow!(e))).await;
            return;
        }
    };
    let mut last_meta = match file.metadata() {
        Ok(m) => m,
        Err(e) => {
            let _ = content_tx.send(Err(anyhow::anyhow!(e))).await;
            return;
        }
    };

    // Read all existing content when the file is first opened and send it.
    if let Err(e) = file.seek(SeekFrom::Start(0)) {
        let _ = content_tx.send(Err(anyhow!(e))).await;
        return;
    }

    // Instead of reading the entire file at once, read it line by line.
    let mut reader = BufReader::new(&file);
    let mut line = String::new();
    while let Ok(bytes_read) = reader.read_line(&mut line) {
        if bytes_read == 0 {
            break; // End of file.
        }

        // Process each line.
        let metadata = process_log_line(&line, &mut graph_resources_log);

        let log_line_info = LogLineInfo { line: line.clone(), metadata };

        let _ = content_tx.send(Ok(log_line_info)).await;
        line.clear(); // Clear for the next line.
    }

    // Reset last_pos to the current EOF.
    let mut last_pos = match file.seek(SeekFrom::End(0)) {
        Ok(p) => p,
        Err(e) => {
            let _ = content_tx.send(Err(anyhow!(e))).await;
            return;
        }
    };

    let mut last_activity = Instant::now();

    loop {
        // Wait for stop or the next check interval.
        tokio::select! {
            _ = &mut stop_rx => {
                eprintln!("Stopping file watcher");
                break;
            },
            _ = tokio::time::sleep(options.check_interval) => {
                // Check if the file has been rotated or truncated.
                match std::fs::metadata(&path) {
                    Ok(meta) => {
                        if !is_same_file(&last_meta, &meta) || meta.len() < last_pos {
                            // The file has been rotated or truncated, reopen it.
                            if let Ok(mut newf) = File::open(&path) {
                                let _ = newf.seek(SeekFrom::Start(0));
                                file = newf;
                                last_meta = meta.clone();
                                last_pos = 0;
                            } else {
                                // Cannot open the new file, wait for the next round.
                                continue;
                            }
                        }
                    }
                    Err(_) => {
                        // Cannot get metadata, skip this round.
                        continue;
                    }
                }

                // If there is new content, read and send it.
                let curr_len = match file.metadata() {
                    Ok(m) => m.len(),
                    Err(_) => continue,
                };

                if curr_len > last_pos {
                    // Read the new part.
                    if let Err(e) = file.seek(SeekFrom::Start(last_pos)) {
                        eprintln!("Error seeking to position: {}", e);
                        let _ = content_tx.send(Err(anyhow::anyhow!(e))).await;
                        break;
                    }

                    let mut reader = BufReader::with_capacity(options.buffer_size, &file);
                    let mut line = String::with_capacity(options.buffer_size);
                    match reader.read_line(&mut line) {
                        Ok(n) if n > 0 => {
                            last_pos += n as u64;

                            // Process the new line.
                            let metadata = process_log_line(&line, &mut graph_resources_log);

                            let log_line_info = LogLineInfo {
                                line,
                                metadata,
                            };
                            let _ = content_tx.send(Ok(log_line_info)).await;
                            last_activity = Instant::now();
                        }
                        Ok(_) => {}
                        Err(e) => {
                            // Specific error for UTF-8 decoding failures.
                            if e.kind() == std::io::ErrorKind::InvalidData {
                                eprintln!("Error: Invalid UTF-8 data in file");
                                let _ = content_tx
                                    .send(Err(anyhow!("Invalid UTF-8 data in file")))
                                    .await;
                            } else {
                                eprintln!("Error reading from file: {}", e);
                                let _ = content_tx.send(Err(anyhow::anyhow!(e))).await;
                            }
                            break;
                        }
                    }
                } else {
                    // Reached EOF, check if the timeout has been reached.
                    if Instant::now().duration_since(last_activity) > options.timeout {
                        eprintln!("Timeout reached, breaking");
                        break;
                    }
                }
            }
        }
    }
}
