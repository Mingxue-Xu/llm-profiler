from src import get_model_config, get_profile

MODEL_ID="meta-llama/Llama-3.2-1B-Instruct"
HF_TOKEN="Your HuggingFace access token here"
BATCH_SIZE=16
TEXT_LENGTH=200
if __name__ == "__main__":

    config=get_model_config(hf_token=HF_TOKEN, model_id=MODEL_ID)
    kwargs = {
        'model_config': config,
        'input_shape': [BATCH_SIZE, TEXT_LENGTH]
    }
    get_profile(kwargs)
    
    # get_profile(kwargs, "flops")
    # get_profile(kwargs, "mult") 
    # get_profile(kwargs, "mult", "attn")
    # get_profile(kwargs, "params", "attn") 
    # get_profile(kwargs, "act")
    # get_profile(kwargs, "act", "mlp") 
