# TK ComfyUI ImageVL

這是一組功能強大的 ComfyUI 自定義節點，專為批量圖片處理及使用 Qwen-VL 系列（Qwen2-VL, Qwen2.5-VL, Qwen3-VL）進行進階視覺語言模型（VLM）反推而設計。

![Workflow Demo](https://github.com/tackcrypto1031/tk_comfyui_imageVL/blob/main/workflow/tk_sample.png)

## 功能特點

- **批量讀取與重新命名圖片**：自動讀取資料夾內的所有圖片，並按順序重新命名（例如：`image_1.png`, `image_2.png`），最後輸出至指定目錄。
- **多模型支援**：
    - **Qwen-VL**：無縫支援 Qwen2-VL、Qwen2.5-VL 及 Qwen3-VL 模型。
    - **JoyCaption**：整合 `fancyfeast/llama-joycaption`，提供高品質的自然語言描述或 Stable Diffusion 標籤。
- **彈性的工作流模式**：
    - **批量處理**：自動處理並儲存整個資料夾的圖片與描述。
    - **單張圖片**：新增獨立節點 (Single)，可直接在工作流中處理單張圖片。
- **進階生成控制**：
    - 可調節 **Max New Tokens**、**解析度控制**、**Temperature** (溫度) 與 **Seed** (種子)。
    - **JoyCaption 專屬**：控制描述類型（描述性、SD Prompt）、長度與語氣。
- **自動存檔**：自動將生成的提示詞儲存為 `.txt` 檔案，檔名與影像完全一致。

## 安裝說明

1.  將此倉庫克隆（Clone）至您的 ComfyUI `custom_nodes` 目錄下：
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/tackcrypto1031/tk_comfyui_imageVL.git
    ```

2.  安裝必要的依賴套件：
    ```bash
    cd tk_comfyui_imageVL
    pip install -r requirements.txt
    ```
    *注意：這需要安裝 `transformers`, `qwen_vl_utils`, `accelerate`, 以及 `huggingface_hub`。*

3.  重新啟動 ComfyUI。

## 使用方法

### 1. 批量圖片讀取 (TK_BatchImageLoader)
此節點負責處理輸入圖片。
- **source_path**：包含原始圖片的目錄。
- **output_path**：重新命名後的圖片儲存目錄。
- **filename_prefix**：檔案名稱前綴（預設為 `image_`）。

### 2. QwenVL 反推節點 (TK_QwenVL_Interrogator)
此節點負責分析圖片並生成描述。
- **model_id**：從下拉選單中選擇所需的 Qwen-VL 模型 (支援 Qwen2, Qwen2.5, Qwen3)。
    - *若本地無模型，系統將自動下載至 `tk_comfyui_imageVL/models`。*
- **prompt**：給模型的指令（例如："請詳細描述這張圖片。"）。
- **min_pixels / max_pixels**：控制視覺編碼器的解析度。
- **max_new_tokens**：生成文字的最大長度。
- **temperature / seed**：生成參數設定。

### 3. TK QwenVL 反推節點 (Single)
處理從其他節點傳入的單張圖片 (IMAGE 類型)。
- **image**：輸入圖片連接。
- **model_id**：選擇 Qwen-VL 模型。
- **prompt**：給模型的指令。
- **Returns**：STRING (生成的影像描述文字)。

### 4. TK JoyCaption 反推節點
JoyCaption 模型的批量處理節點，專為自然語言描述或 SD 提示詞設計。
- **joycaption_model**：選擇模型 (例如：`fancyfeast/llama-joycaption-beta-one-hf-llava`)。
- **caption_type**：
    - `Descriptive`：正式、自然語言的詳細描述。
    - `Stable Diffusion Prompt`：以逗號分隔的標籤格式，包含品質修飾詞。
- **caption_length**：限制輸出長度 (極短 - 極長)。
- **user_prompt**：使用自定義指令覆蓋內建的系統指令。
- **cache_model**：保持模型載入狀態（建議批量處理時開啟）。

### 5. TK JoyCaption 反推節點 (Single)
JoyCaption 的單張圖片版本。
- **image**：輸入圖片連接。
- **joycaption_model**：選擇模型。
- **caption_type / caption_length**：格式控制。
- **Returns**：STRING (生成的影像描述文字)。

### 6. 文字儲存節點 (TK_TextSaver)
此節點僅為工作流相容性保留。文字儲存功能現已**自動由反推節點處理** (TK QwenVL Interrogator / TK JoyCaption Interrogator)，系統會在 `output_path` 自動儲存與圖片同名的 .txt 檔案。

## 工作流範例

1.  **載入圖片**：將 `TK Batch Image Loader` 連接至 `TK QwenVL Interrogator`。
2.  **生成描述**：將 `TK QwenVL Interrogator` (texts, filenames) 連接至 `TK Text Saver`。
3.  **執行任務**：點擊 "Queue Prompt"，系統將自動循環處理資料夾內的所有圖片。

## 致謝

基於 [Qwen-VL](https://github.com/QwenLM/Qwen-VL) 架構開發。
靈感來自 ComfyUI 社群的多項貢獻。
