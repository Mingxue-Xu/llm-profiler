from src import get_model_config, get_profile

MODEL_ID="meta-llama/Llama-3.2-1B-Instruct"           
HF_TOKEN="your hf token"
BATCH_SIZE=16
TEXT_LENGTH=400
if __name__ == "__main__":

    config=get_model_config(hf_token=HF_TOKEN, model_id=MODEL_ID)
    kwargs = {
        'model_config': config,
        'input_shape': [BATCH_SIZE, TEXT_LENGTH]
    }
    flopsunit = get_profile(kwargs=kwargs)
    # flopsunit = get_profile(kwargs=kwargs, attr='flops')
    # get_profile(kwargs, 'act', 'mlp')

