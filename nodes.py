
import os
import shutil
import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from huggingface_hub import snapshot_download

class TK_BatchImageLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_path": ("STRING", {"default": "C:/input_images"}),
                "output_path": ("STRING", {"default": "C:/output_images"}),
                "filename_prefix": ("STRING", {"default": "image_"}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("image_paths",)
    FUNCTION = "process_images"
    CATEGORY = "TK/Image"

    def process_images(self, source_path, output_path, filename_prefix):
        if not os.path.exists(source_path):
            print(f"Source path {source_path} does not exist.")
            return ([],)
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Get all valid image files
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [f for f in os.listdir(source_path) if os.path.splitext(f)[1].lower() in valid_extensions]
        files.sort() # Ensure consistent order

        processed_paths = []

        for idx, filename in enumerate(files, start=1):
            src_file_path = os.path.join(source_path, filename)
            ext = os.path.splitext(filename)[1]
            new_filename = f"{filename_prefix}{idx}{ext}"
            dst_file_path = os.path.join(output_path, new_filename)

            # Copy file to output directory with new name
            shutil.copy2(src_file_path, dst_file_path)
            processed_paths.append(dst_file_path)

        return (processed_paths,)


class TK_QwenVL_Interrogator:
    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model_id = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_paths": ("LIST",),
                "model_id": ([
                    "Qwen/Qwen2.5-VL-7B-Instruct",
                    "Qwen/Qwen2.5-VL-3B-Instruct",
                    "Qwen/Qwen2.5-VL-72B-Instruct",
                    "Qwen/Qwen2-VL-7B-Instruct",
                    "Qwen/Qwen2-VL-2B-Instruct",
                    "Qwen/Qwen2-VL-72B-Instruct",
                    # Qwen3-VL (Assuming naming convention or placeholders until release if not fully out on HF main hub yet, 
                    # but based on search results the ecosystem is ready or they are under Qwen/ collection with specific names)
                    # Adding specific Qwen3 names found in search or placeholders if they are cutting edge:
                    "Qwen/Qwen3-VL-236B-Instruct", # From search
                    "Qwen/Qwen3-VL-30B-Instruct",  # From search, using likely main naming
                    "Qwen/Qwen3-VL-8B-Instruct",
                    "Qwen/Qwen3-VL-4B-Instruct",
                    "Qwen/Qwen3-VL-2B-Instruct",
                 ], {"default": "Qwen/Qwen2.5-VL-7B-Instruct"}),
                "prompt": ("STRING", {"default": "Describe this image.", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192, "step": 1}),
                "min_pixels": ("INT", {"default": 200704, "min": 1024, "max": 99999999, "step": 1}),
                "max_pixels": ("INT", {"default": 1003520, "min": 1024, "max": 99999999, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("LIST", "LIST")
    RETURN_NAMES = ("texts", "filenames")
    FUNCTION = "interrogate"
    CATEGORY = "TK/QwenVL"

    def interrogate(self, image_paths, model_id, prompt, max_new_tokens, min_pixels, max_pixels, temperature, seed):
        # Set seed for reproducibility
        if seed is not None:
             torch.manual_seed(seed)

        # Determine model directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "models")
        
        # Sanitize model_id to create a valid folder name (replaces / with __)
        model_folder_name = model_id.replace("/", "__")
        model_path = os.path.join(models_dir, model_folder_name)

        # Check if model exists, if not download
        if not os.path.exists(model_path):
            print(f"Model {model_id} not found at {model_path}. Downloading...")
            try:
                snapshot_download(repo_id=model_id, local_dir=model_path)
                print(f"Model downloaded to {model_path}")
            except Exception as e:
                print(f"Failed to download model: {e}")
                # Fallback or re-raise? Re-raising to alert user.
                raise e

        # Load model if strictly necessary (different ID or not loaded)
        if self.model is None or self.current_model_id != model_id:
            print(f"Loading model from {model_path}...")
            # Use AutoModelForVision2Seq to automatically pick the correct class (e.g. Qwen2VL, Qwen2.5, Qwen3)
            # based on the config.json in the model folder.
            from transformers import AutoModelForVision2Seq

            try:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_path, 
                    torch_dtype="auto", 
                    device_map="auto",
                    trust_remote_code=True # Enabled to support new/custom architectures if needed
                )
                self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                self.current_model_id = model_id
            except Exception as e:
                 print(f"Error loading model: {e}")
                 raise e

        generated_texts = []
        filenames = []

        for img_path in image_paths:
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img_path,
                            "min_pixels": min_pixels,
                            "max_pixels": max_pixels,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            # Generate parameters
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": True if temperature > 0 else False,
            }
            if seed is not None and temperature > 0:
                # Transformers generate usually takes seed via setting torch manual seed which we did top level, 
                # but explicit passing isn't standard in generate() kwargs for all models. rely on global seed.
                pass

            generated_ids = self.model.generate(**inputs, **gen_kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            generated_texts.append(output_text)
            
            # Extract filename for the parallel list
            basename = os.path.basename(img_path)
            filenames.append(basename)

        return (generated_texts, filenames)

class TK_TextSaver:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "texts": ("LIST",),
                "filenames": ("LIST",),
                "output_path": ("STRING", {"default": "C:/output_text"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_texts"
    CATEGORY = "TK/Text"
    OUTPUT_NODE = True

    def save_texts(self, texts, filenames, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for text, filename in zip(texts, filenames):
            # Replace extension with .txt
            name_only = os.path.splitext(filename)[0]
            txt_filename = f"{name_only}.txt"
            save_path = os.path.join(output_path, txt_filename)

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(text)

        return {}
