
import os
import shutil
import torch
import traceback
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import folder_paths
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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("LIST", "LIST")
    RETURN_NAMES = ("texts", "filenames")
    FUNCTION = "interrogate"
    CATEGORY = "TK/QwenVL"

    def interrogate(self, image_paths, model_id, prompt, max_new_tokens, min_pixels, max_pixels, temperature, seed, save_per_image=True):
        # Always enforce save_per_image = True to prevent data loss, even if passed otherwise (though it won't be passed from UI)
        save_per_image = True 
        
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

            try:
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
                
                generated_ids = self.model.generate(**inputs, **gen_kwargs)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                # Immediate saving logic
                if save_per_image:
                    try:
                        # Save to image directory
                        target_dir = os.path.dirname(img_path)

                        basename = os.path.basename(img_path)
                        name_only = os.path.splitext(basename)[0]
                        txt_filename = f"{name_only}.txt"
                        save_path = os.path.join(target_dir, txt_filename)
                        
                        with open(save_path, "w", encoding="utf-8") as f:
                            f.write(output_text)
                            
                        print(f"Saved caption to: {save_path}")
                    except Exception as e:
                        print(f"Error saving file {save_path}: {e}")
                        traceback.print_exc()

                generated_texts.append(output_text)
                
                # Extract filename for the parallel list
                basename = os.path.basename(img_path)
                filenames.append(basename)

            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                traceback.print_exc()
                generated_texts.append("") # Keep list length consistent
                filenames.append(os.path.basename(img_path))
            
            # Clear cache
            torch.cuda.empty_cache()

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
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_texts"
    CATEGORY = "TK/Text"
    OUTPUT_NODE = True

    def save_texts(self, texts, filenames):
        # Text saving is now handled by the upstream Interrogator node.
        # This node remains for workflow compatibility but performs no action.
        return {}


class TK_JoyCaption_Interrogator:
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_model_id = None
        self.image_adapter = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_path": ("STRING", {"default": "C:/input_images"}),
                "output_path": ("STRING", {"default": "C:/output_images"}),
                "joycaption_model": ([
                    "fancyfeast/llama-joycaption-beta-one-hf-llava",
                    "fancyfeast/joy-caption-pre-alpha"
                ], {"default": "fancyfeast/llama-joycaption-beta-one-hf-llava"}),
                 "caption_type": ([
                    "Descriptive",
                    "Stable Diffusion Prompt",
                ], {"default": "Descriptive"}),
                "caption_length": ([
                    "any", 
                    "very short", 
                    "short", 
                    "medium-length", 
                    "long", 
                    "very long"
                ], {"default": "long"}),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 100}),
                "cache_model": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "image_"}),
            },
        }

    RETURN_TYPES = ("LIST", "LIST")
    RETURN_NAMES = ("texts", "filenames")
    FUNCTION = "interrogate"
    CATEGORY = "TK/JoyCaption"

    def interrogate(self, source_path, output_path, joycaption_model, caption_type, caption_length, user_prompt, max_new_tokens, temperature, top_p, top_k, cache_model, filename_prefix):
        
        # 1. Prepare Model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "models")
        model_folder_name = joycaption_model.replace("/", "__")
        model_path = os.path.join(models_dir, model_folder_name)

        if not os.path.exists(model_path):
            print(f"Model {joycaption_model} not found at {model_path}. Downloading...")
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=joycaption_model, local_dir=model_path)
                print(f"Model downloaded to {model_path}")
            except Exception as e:
                print(f"Failed to download model: {e}")
                raise e

        # Load Model (Singleton logic within instance for now, or global cache if needed across nodes)
        if self.model is None or self.current_model_id != joycaption_model:
            print(f"Loading model {joycaption_model}...")
            
            try:
                # Import here to avoid global dependency issues if unused
                from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoModelForCausalLM
                
                # Check for Beta One (LLaVA based) vs Pre-Alpha (Custom Adapter)
                if "beta-one" in joycaption_model:
                    # LLaVA style loading - Use AutoModelForVision2Seq for VLMs
                    from transformers import AutoModelForVision2Seq
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        model_path, 
                        torch_dtype="auto", 
                        device_map="auto",
                        trust_remote_code=True
                    )
                    self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                    self.image_adapter = None # Integrated
                else:
                    # Pre-Alpha / Custom Adapter Logic
                    # This path might require specific separate loading of SigLIP + Adapter + Llama
                    # For now, implementing basic load assuming it's a unified HF repo or similar structure
                    # If it's the split structure, we might need more complex logic.
                    # Assuming the user selected the "merged" one or compatible one.
                    from transformers import AutoModelForCausalLM
                    self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True)
                    self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

                self.current_model_id = joycaption_model
                
            except Exception as e:
                print(f"Error loading model: {e}")
                raise e

        # 2. Process Files
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [f for f in os.listdir(source_path) if os.path.splitext(f)[1].lower() in valid_extensions]
        files.sort()
        
        generated_texts = []
        filenames = []

        # Prompt Construction
        base_prompt = user_prompt
        if not base_prompt:
             # Default prompts based on caption_type
            prompts = {
                "Descriptive": "Write a descriptive caption for this image in a formal tone.",
                "Descriptive": "Write a descriptive caption for this image in a formal tone.",
                "Stable Diffusion Prompt": "Write a Stable Diffusion prompt for this image. Start with quality tags (e.g., masterpiece, best quality, 4k). Use a tag-based format separated by commas. Describe the subject, action, context, and art style.",
            }
            base_prompt = prompts.get(caption_type, "Write a descriptive caption for this image.")
            
        if caption_length and caption_length != "any":
            base_prompt += f" Keep it {caption_length}."

        print(f"Starting Generation with prompt: {base_prompt}")

        import PIL.Image
        
        for idx, filename in enumerate(files, start=1):
            img_path = os.path.join(source_path, filename)
            try:
                image = PIL.Image.open(img_path)
                
                # Inference
                # Prepare inputs
                conversation = [
                    {
                        "role": "user",
                        "content": base_prompt, # Simplest form usually works with LLaVA processors if images are passed separately
                    },
                ]
                
                # Apply template
                text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                
                # Process inputs
                inputs = self.processor(text=text_prompt, images=image, return_tensors="pt")
                inputs = inputs.to(self.model.device)
                
                # Generate
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "do_sample": True if temperature > 0 else False,
                }
                
                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, **gen_kwargs)
                
                # Decode
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                # Save
                ext = os.path.splitext(filename)[1]
                
                new_image_filename = f"{filename_prefix}{idx}{ext}"
                new_text_filename = f"{filename_prefix}{idx}.txt"

                save_image_path = os.path.join(output_path, new_image_filename)
                save_text_path = os.path.join(output_path, new_text_filename)
                
                # Copy Image
                shutil.copy2(img_path, save_image_path)

                with open(save_text_path, "w", encoding="utf-8") as f:
                    f.write(output_text)
                
                print(f"Saved: {save_text_path}")
                generated_texts.append(output_text)
                filenames.append(new_image_filename)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                traceback.print_exc()
                generated_texts.append("")
                filenames.append(filename)

        if not cache_model:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            torch.cuda.empty_cache()

        return (generated_texts, filenames)

NODE_CLASS_MAPPINGS = {
    "TK_BatchImageLoader": TK_BatchImageLoader,
    "TK_QwenVL_Interrogator": TK_QwenVL_Interrogator,
    "TK_TextSaver": TK_TextSaver,
    "TK_JoyCaption_Interrogator": TK_JoyCaption_Interrogator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TK_BatchImageLoader": "TK Batch Image Loader",
    "TK_QwenVL_Interrogator": "TK QwenVL Interrogator",
    "TK_TextSaver": "TK Text Saver",
    "TK_JoyCaption_Interrogator": "TK JoyCaption Interrogator",
}

