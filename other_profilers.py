import torch
from torch.profiler import _KinetoProfile, profile
from transformers import AutoTokenizer,AutoModelForCausalLM, LlamaForCausalLM, LlamaModel, LlamaConfig
MODEL_ID="meta-llama/Llama-3.2-1B-Instruct"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE)
input_tensor = torch.ones([2, 4], dtype=torch.int32, device=DEVICE)
# kineto = _KinetoProfile(
#     record_shapes=True,
#     profile_memory=True,
#     with_flops=True,
#     with_modules=True,
#     acc_events=True,
# )
# kineto.start()
# model(input_tensor)
# kineto.stop()

profiler = profile(
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
    with_modules=True,
    acc_events=True,
    use_cuda=True,
)
profiler.start()
model(input_tensor)
profiler.stop()

results = profiler.profiler.function_events
results_t = results[2500:2510]