# TK ComfyUI ImageVL

A powerful set of ComfyUI custom nodes designed for batch image processing and advanced Vision-Language Model (VLM) interrogation using the Qwen-VL family (Qwen2-VL, Qwen2.5-VL, Qwen3-VL).

![Workflow Demo](https://github.com/tackcrypto1031/tk_comfyui_imageVL/blob/main/workflow/tk_sample.png)

## Features

- **Batch Image Loading & Renaming**: Automatically read all images from a folder, rename them sequentially (e.g., `image_1.png`, `image_2.png`), and save them to a specified output directory.
- **Qwen-VL Integration**: Seamless support for Qwen-VL models with automatic downloading from Hugging Face.
    - Support for **Qwen2-VL**, **Qwen2.5-VL**, and **Qwen3-VL**.
    - Auto-detection of model architecture.
- **Advanced Generation Control**:
    - Adjustable **Max New Tokens** for longer, detailed descriptions.
    - **Resolution Control** (Min/Max Pixels) to balance performance and detail.
    - **Temperature** and **Seed** control for reproducible results.
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
- **model_id**: Select the desired Qwen-VL model from the dropdown. 
    - *Models will be automatically downloaded to `tk_comfyui_imageVL/models` if not present.*
- **prompt**: The instruction for the model (e.g., "Describe this image detailedly.").
- **min_pixels / max_pixels**: Control the resolution for the vision encoder.
- **max_new_tokens**: Maximum length of the generated text.
- **temperature / seed**: Generation parameters.

### 3. Text Saver (TK_TextSaver)
This node saves the results.
- **output_path**: Directory where the text files will be saved (filename matches the image).

## Workflow Example

1.  **Load Images**: Connect `TK Batch Image Loader` to `TK QwenVL Interrogator`.
2.  **Generate**: Connect `TK QwenVL Interrogator` (texts, filenames) to `TK Text Saver`.
3.  **Run**: Press "Queue Prompt" to process the entire folder in batch.

## Credits

Based on the [Qwen-VL](https://github.com/QwenLM/Qwen-VL) architecture.
Inspired by various ComfyUI community contributions.
