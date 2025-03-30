from transformers.models.llama import LlamaConfig

def get_llama_config(hf_token: str, model_id: str):
    try:
        config = LlamaConfig.from_pretrained(token=hf_token, pretrained_model_name_or_path=model_id)
    except OSError:
        print(f"LlamaConfig not found for {model_id}")
        return

    return config

def get_model_config(hf_token: str, model_id: str):
    if "llama" in model_id.lower():
        return get_llama_config(hf_token=hf_token, model_id=model_id)
    else:
        raise NotImplementedError(f"{model_id} is not supported")
        return


