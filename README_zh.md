# TK ComfyUI ImageVL

這是一組功能強大的 ComfyUI 自定義節點，專為批量圖片處理及使用 Qwen-VL 系列（Qwen2-VL, Qwen2.5-VL, Qwen3-VL）進行進階視覺語言模型（VLM）反推而設計。

![Workflow Demo](https://github.com/tackcrypto1031/tk_comfyui_imageVL/blob/main/workflow/tk_sample.png)

## 功能特點

- **批量讀取與重新命名圖片**：自動讀取資料夾內的所有圖片，並按順序重新命名（例如：`image_1.png`, `image_2.png`），最後輸出至指定目錄。
- **Qwen-VL 模型整合**：完美支援 Qwen-VL 模型，並具備從 Hugging Face 自動下載模型的功能。
    - 支援 **Qwen2-VL**、**Qwen2.5-VL** 及 **Qwen3-VL**。
    - 自動偵測模型架構。
- **進階生成控制**：
    - 可調節 **Max New Tokens** 以獲得更長、更詳細的描述。
    - **解析度控制** (Min/Max Pixels) 以平衡效能與細節。
    - 支援 **Temperature** (溫度) 與 **Seed** (種子) 控制，確保結果的可重複性。
- **自動儲存**：自動將生成的提示詞儲存為 `.txt` 檔案，檔名與原始圖片完全一致。

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
- **model_id**：從下拉選單中選擇所需的 Qwen-VL 模型。
    - *若本地無模型，系統將自動下載至 `tk_comfyui_imageVL/models`。*
- **prompt**：給模型的指令（例如："請詳細描述這張圖片。"）。
- **min_pixels / max_pixels**：控制視覺編碼器的解析度。
- **max_new_tokens**：生成文字的最大長度。
- **temperature / seed**：生成參數設定。

### 3. 文字儲存節點 (TK_TextSaver)
此節點負責儲存生成的結果。
- **output_path**：文字檔案儲存目錄（檔名將與圖片匹配）。

## 工作流範例

1.  **載入圖片**：將 `TK Batch Image Loader` 連接至 `TK QwenVL Interrogator`。
2.  **生成描述**：將 `TK QwenVL Interrogator` (texts, filenames) 連接至 `TK Text Saver`。
3.  **執行任務**：點擊 "Queue Prompt"，系統將自動循環處理資料夾內的所有圖片。

## 致謝

基於 [Qwen-VL](https://github.com/QwenLM/Qwen-VL) 架構開發。
靈感來自 ComfyUI 社群的多項貢獻。
