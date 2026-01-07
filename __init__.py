from .nodes import TK_BatchImageLoader, TK_QwenVL_Interrogator, TK_TextSaver

NODE_CLASS_MAPPINGS = {
    "TK_BatchImageLoader": TK_BatchImageLoader,
    "TK_QwenVL_Interrogator": TK_QwenVL_Interrogator,
    "TK_TextSaver": TK_TextSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TK_BatchImageLoader": "TK Batch Image Loader",
    "TK_QwenVL_Interrogator": "TK QwenVL Interrogator",
    "TK_TextSaver": "TK Text Saver"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
