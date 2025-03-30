
# LLM Profiler
A fine-grained layer-level LLM profiler that calculates the algorithmic operations (e.g. ADD, MULT), parameters, cache (buffers) and activations.

## Setup Environment
We use `virualenv` in this repository. To set up the python environment, in the terminal,
```shell
$ virtualenv env
$ source env/bin/activate
(env) $ pip install -r requirements.txt
```
## Quick Start
In the root directory of this repo,
```shell
(env) $ python3 simple_test.py
```
The output in the terminal should be
```shell
transformer with input batchsize 16 and sequence length 200:
    ADD:        58.28   (10^9)
    MULT:       59.02   (10^9)
    DIV:        327.89  (10^6)
    SQRT:       16
    PARAMS:     15.83   (10^9)
    CACHE:      59.10   (10^6)
    ACT:        15.58   (10^9)
```
The following instruction is based on the structure of `simple_test.py`.

## Instruction
### Preparation
Give the model name (`MODEL_ID`, which should be available on [HuggingFace](https://huggingface.co/)) and the input batchsize `BATCH_SIZE` and maximum sequence length `TEXT_LENGTH`. If you don't have HuggingFace access token, create one according to [here](https://huggingface.co/docs/hub/security-tokens).
```python
from src import get_model_config, get_profile

MODEL_ID="meta-llama/Llama-3.2-1B-Instruct" 
HF_TOKEN="Your HuggingFace access token here" 
BATCH_SIZE=16
TEXT_LENGTH=200
```
### Get the fine-grained FLOPs of a single forward pass
```python
model_config = get_model_config(model_id = MODEL_ID, hf_token = HF_TOKEN)
kwargs =    { 
                'model_config': model_config, 
                'input_shape':  [BATCH_SIZE,TEXT_LEN]
            }   
```
#### Get the total transformer FLOPs (including ADD and MULT) during the inference (a single forwarding pass):
```python
get_profile(kwargs, "flops")
```
The output in the terminal should be something like
```shell
transformer with input batchsize 16 and sequence length 200:
    FLOPs (ADD and MULT):      117.30  (10^9)
```
#### Get the total transformer element-wise multiplication operations during the inference
```python                   
get_profile(kwargs, "mult")                   
```
The output in the terminal should be something like
```shell
transformer with input batchsize 16 and sequence length 200:
    MULT:       59.02  (10^9)
```
#### Get the element-wise multiplication operations involved in the *all attention layers*
```python       
get_profile(kwargs, "mult", "attn")         
```
The output in the terminal should be something like
```shell
attn with input batchsize 16 and sequence length 200:
    MULT:       45.22  (10^9)
```

#### Get the model parameters involved in the *all attention layers*
```python   
get_profile(kwargs, "params", "attn")       
```
The output in the terminal should be something like
```shell
attn with input batchsize 16 and sequence length 200:
    PARAMS:     2.68  (10^9)
```
#### Get the activations (intermediate output between the layers) of the transformer
```python   
get_profile(kwargs, "act")                
```
The output in the terminal should be something like
```shell
transformer with input batchsize 16 and sequence length 200:
    ACT:        15.58  (10^9)
```
#### Get the activations (intermediate output between the layers) of the *all the mlp layers*
```python   
get_profile(kwargs, "act", "mlp")        
```
The output in the terminal should be something like
```shell
mlp with input batchsize 16 and sequence length 200:
    ACT:        12.88   (10^9)
```

For how to set `kwargs` and what kind of data are available, please refer to [docs/kwargs](docs/kwargs.md)

## Detailed Explanation

### What are considered and not considered in `llm-profiler`?
**Arithmetic Operations** that are considered:
- `ADD`: element-wise addition
- `MULT`: element-wise multiplication
- `DIV`: vector or tensor division, can be implemented differently according to hardware architectures, etc
- `SQRT`: vector or tensor square root, can be implemented differently according to the computing systems, etc
- `FLOPs`: element-wise addition and multiplication

**Dynamics** that are considered:
- `PARAMS`: the layers' parameters or the transformer parameters
- `ACT`: activations (intermediate output between the layers) of the layers or the transformer
- `CACHE`: temorary buffers of the layers or the transformer  

NOT considered:
- logistic sigmoid (in activation functions)
- `output_hidden_states=True` or `output_attentions=True` in the transformer forward pass (e.g. in [`LlamaModel.forward`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L518))
- dynamic rotary embedding (currently only [`llama3 rotary embedding`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py#L322) is supported)

Plan to consider:
- [] self-defined Model Configuration
- [] other LLM architecture (e.g. Gemma)
- [] other scope like `torch.nn.Linear` and `torch.nn.Module.named_parameter` (e.g. `model.layers.12.self_attn.k_proj` and `model.layers.1.mlp.up_proj`)



**NOTE**: For detailed reference code/explanation for FLOPs calculation, a sample reference soure code is in [docs/flops](docs/flops.md).

## Differences between `llm-profiler` and other profilers

The emphasis of `llm-profiler` is **layer-level**, as well as fine-grained algorithmic operations, rather than simplely use FLOPs to desceibe. We separate intermediate output between the layers, temporay buffer and parameters, while others mainly focus on overall operations and system peak memory.

Other profilers are:

### [DeepSpeed](https://github.com/deepspeedai/DeepSpeed/tree/master/deepspeed/profiling/flops_profiler)
Gives FLOPs per layer, however, doesn't distinguish addition and summation, wich can be very different (latency \& energy consumption) when LLMs deployed on different devices.

### [PyTorch Profiler](https://github.com/pytorch/kineto) and [`torch.autograd.profiler`](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/autograd/profiler.py) 
Coarse-grained profiler that gives the FLOPs and memory, CPU usage, etc., on the whole model level, rather than layer-level.

### [Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis)
A more visualized, systematic model-level profiler based on [PyTorch Profiler](https://github.com/pytorch/kineto). 