//
// Copyright © 2025 Agora
// This file is part of TEN Framework, an open source project.
// Licensed under the Apache License, Version 2.0, with certain conditions.
// Refer to the "LICENSE" file in the root directory for more information.
//
use actix_web_actors::ws::WebsocketContext;

use ten_rust::pkg_info::manifest::support::ManifestSupport;

use super::common::run_installation;
use super::WsBuiltinFunction;
use crate::cmd::cmd_install::LocalInstallMode;

impl WsBuiltinFunction {
    pub fn install(
        &mut self,
        base_dir: String,
        pkg_type: String,
        pkg_name: String,
        pkg_version: Option<String>,
        ctx: &mut WebsocketContext<WsBuiltinFunction>,
    ) {
        let pkg_name_and_version = if let Some(pkg_version) = pkg_version {
            format!("{}@{}", pkg_name, pkg_version)
        } else {
            pkg_name
        };

        let install_command = crate::cmd::cmd_install::InstallCommand {
            package_type: Some(pkg_type),
            package_name: Some(pkg_name_and_version),
            support: ManifestSupport { os: None, arch: None },
            local_install_mode: LocalInstallMode::Link,
            standalone: false,
            local_path: None,
            cwd: base_dir.clone(),
        };

        run_installation(
            self.tman_config.clone(),
            self.tman_metadata.clone(),
            install_command,
            ctx,
        );
    }
}
