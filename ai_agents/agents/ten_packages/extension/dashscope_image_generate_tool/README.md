# Image Generate Tool

A TEN extension for generating images using Alibaba Cloud's Text-to-Image service.

## Features

- Generate images from text descriptions using Alibaba Cloud's Text-to-Image API
- Support for customizable image size, number of images, and other parameters
- Async operation support

## API

### Configuration Properties

- `api_key`: Alibaba Cloud API key
- `model`: Model name (default: "wanx2.1-t2i-turbo")
- `size`: Image size (default: "1024*1024")
- `n`: Number of images to generate (default: 1)
- `prompt_extend`: Enable prompt extension (default: true)
- `watermark`: Add watermark to images (default: false)

### Tool Functions

- `image_generate`: Generate images from text prompt
  - Parameters:
    - `prompt`: Text description for image generation

## Development

### Dependencies

- dashscope

### Build

No special build steps required. Install dependencies using:
```bash
pip install -r requirements.txt
```

### Testing

Test the extension by providing valid API credentials and running the image generation tool.

## Misc

<!-- others if applicable -->
