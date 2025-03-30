#   How to set Model and Model Input in `llm-profiler`
Model and model input are set in a python `Dict` named `kwargs`.
## Hardware independent LLM Configuration
Arithmetic operations and model parameters are only determined on LLM architecture rather than what kind of devices the LLM is deployed on. 
For this case, model configuration is enough for the profiler.
The model configuration should be the type of `transformers.PretrainedConfig`, or its subclasses for specific LLM architectures like `transformers.LlamaConfig`. To get such a configuration,
```python
from llm_profiler import get_model_config
model_config = get_model_config(model_id = "meta-llama/Llama-3.2-3B-Instruct", \
                hf_token = "your HF Access token")
```
or use the classes in `transformers`,
```python
import transformers
model_config = transformers.AutoConfig(pretrained_model_name_or_path = "meta-llama/Llama-3.2-3B-Instruct", \
                token = "your HF Access token")
#  or for specific models
model_config = transformers.LlamaConfig(pretrained_model_name_or_path = "meta-llama/Llama-3.2-3B-Instruct", \
                token = "your HF Access token")
```


When the model configuration is successfully loaded, put it into `kwargs`
```python
kwargs = {
    ...
    'model_config': model_config,
    ...
}
```


## Model Input Tensor Dimensions
The default model input is a 2-dimentional tensor, which can be assigned with a tensor shape. The first dimension of the tensor is batch size, and the second is the seqence length (or the maximum text length of the input batch). For example,
```python
kwargs = {
    ...
    'input_shape': [16,100],
    ...
}
```
means the input tensor has a batch size 16, and the maximum text length is 100.

#   What Kind of Profiling Data are Available in `llm-profiler`
The interface to get the profiling data is
```python
get_profile(    
    kwargs:Dict,
    scope:Optional[AnyStr]=None,
    attr:Optional[AnyStr]=None
)     
```

## `scope`: the scope for profiling

The supported scope in `llm-profiler` are as follows:
1. model-level 
2. layer-level
    - defined layers in `transformer` (e.g. *LlamaAttention*, *LlamaDecoderLayer*)
    - defined layers in `torch.nn` (e.g. *Linear*, *LlamaDecoderLayer*)

All of these can be assigned via `scope`. There are several possible values of `scope`:
-   `transformer` or not given: the calculated scope is the whole transformer module (layers for downstream tasks are not included);
-   `attn`: all the attention layers (e.g. *LlamaAttention*) in the transformer blocks of the model;
-  `mlp`: all the MLP layers (e.g. *LlamaMLP*) in the transformer blocks of the model;

## `attr`: profiled values of interest
The supported profiled values in `llm-profiler` are as follows:
- `FLOPS`: total FLOPs (addition and multiplication)
- `ADD`: element-wise addition
- `MULT`: element-wise multiplication
- `DIV`: appearance of division
- `SQRT`: appearance of square root
- `PARAMS`: parameters
- `ACT`: activations (intermediate output) during the single forward pass
- `CACHE`: temperary beffers that will be released before the model gots the final output
