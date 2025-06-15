from src.models.model_base import ModelBase

def construct_model_base(model_path: str) -> ModelBase:
    if 'qwen' in model_path.lower():
        from src.models.qwen2_model import Qwen2Model
        return Qwen2Model(model_path)
    elif 'llama-3' in model_path.lower():
        from src.models.llama3_model import Llama3Model
        return Llama3Model(model_path)
    elif 'gemma-2-' in model_path.lower():
        from src.models.gemma2_model import Gemma2Model
        return Gemma2Model(model_path)
    else:
        raise ValueError(f"Unsupported model family: {model_path}")
