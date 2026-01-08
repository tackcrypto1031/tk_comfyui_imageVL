# TK ComfyUI ImageVL

[中文說明文件 (README_zh.md)](https://github.com/tackcrypto1031/tk_comfyui_imageVL/blob/main/README_zh.md)

A powerful set of ComfyUI custom nodes designed for batch image processing and advanced Vision-Language Model (VLM) interrogation using the Qwen-VL family (Qwen2-VL, Qwen2.5-VL, Qwen3-VL).

![Workflow Demo](https://github.com/tackcrypto1031/tk_comfyui_imageVL/blob/main/workflow/tk_sample.png)

## Features

- **Batch Image Loading & Renaming**: Automatically read all images from a folder, rename them sequentially (e.g., `image_1.png`, `image_2.png`), and save them to a specified output directory.
- **Multi-Model Support**:
    - **Qwen-VL**: Seamless support for Qwen2-VL, Qwen2.5-VL, and Qwen3-VL models.
    - **JoyCaption**: Integration with `fancyfeast/llama-joycaption` for high-quality, natural language captions or Stable Diffusion tags.
- **Flexible Workflow Modes**:
    - **Batch Processing**: Process entire directories of images with auto-saving.
    - **Single Image**: New independent nodes (Single) to process individual images directly from your workflow.
- **Advanced Generation Control**:
    - Adjustable **Max New Tokens**, **Resolution Control**, **Temperature**, and **Seed**.
    - **JoyCaption specific**: Control caption type ('Descriptive', 'SD Prompt'), length, and tone.
- **Auto-Saving**: Automatically saves the generated captions as `.txt` files matching the image filenames.

## Installation

1.  Clone this repository into your ComfyUI `custom_nodes` directory:
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/tackcrypto1031/tk_comfyui_imageVL.git
    ```

2.  Install the required dependencies:
    ```bash
    cd tk_comfyui_imageVL
    pip install -r requirements.txt
    ```
    *Note: This requires `transformers`, `qwen_vl_utils`, `accelerate`, and `huggingface_hub`.*

3.  Restart ComfyUI.

## Usage

### 1. Batch Image Loader (TK_BatchImageLoader)
This node handles the input images.
- **source_path**: Directory containing your original images.
- **output_path**: Directory where renamed images will be saved.
- **filename_prefix**: Prefix for the renamed files (default: `image_`).

### 2. QwenVL Interrogator (TK_QwenVL_Interrogator)
This node analyzes the images and generates descriptions.
- **model_id**: Select the desired Qwen-VL model from the dropdown (Supports Qwen2, Qwen2.5, Qwen3). 
    - *Models will be automatically downloaded to `tk_comfyui_imageVL/models` if not present.*
- **prompt**: The instruction for the model (e.g., "Describe this image detailedly.").
- **min_pixels / max_pixels**: Control the resolution for the vision encoder.
- **max_new_tokens**: Maximum length of the generated text.
- **temperature / seed**: Generation parameters.

### 3. TK QwenVL Interrogator (Single)
Process a single image passed directly from another node (IMAGE type).
- **image**: Input image connection.
- **model_id**: Select Qwen-VL model.
- **prompt**: Instruction for the model.
- **Returns**: STRING (generated text).

### 4. TK JoyCaption Interrogator
Batch processing node for JoyCaption models, designed for natural language captions or SD prompts.
- **joycaption_model**: Select model (e.g., `fancyfeast/llama-joycaption-beta-one-hf-llava`).
- **caption_type**: 
    - `Descriptive`: Formal, natural language description.
    - `Stable Diffusion Prompt`: Tag-based format with quality boosters.
- **caption_length**: constrain the output length (very, short - very long).
- **user_prompt**: Override the internal system prompt with your own instruction.
- **cache_model**: Keep model loaded (recommended for batch).

### 5. TK JoyCaption Interrogator (Single)
Single image version of JoyCaption.
- **image**: Input image connection.
- **joycaption_model**: Select model.
- **caption_type / caption_length**: Format controls.
- **Returns**: STRING (generated text).

### 6. Text Saver (TK_TextSaver)
This node is kept for workflow compatibility. Text saving is now **handled automatically** by the Interrogator nodes (TK QwenVL Interrogator / TK JoyCaption Interrogator), which save a .txt file alongside the processed image in the `output_path`.

## Workflow Example

1.  **Load Images**: Connect `TK Batch Image Loader` to `TK QwenVL Interrogator`.
2.  **Generate**: Connect `TK QwenVL Interrogator` (texts, filenames) to `TK Text Saver`.
3.  **Run**: Press "Queue Prompt" to process the entire folder in batch.

## Credits

Based on the [Qwen-VL](https://github.com/QwenLM/Qwen-VL) architecture.
Inspired by various ComfyUI community contributions.
