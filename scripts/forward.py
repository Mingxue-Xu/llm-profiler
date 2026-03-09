from transformers import AutoTokenizer,AutoModelForCausalLM, LlamaForCausalLM, LlamaModel, LlamaConfig

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm, LlamaAttention, eager_attention_forward, ALL_ATTENTION_FUNCTIONS, LlamaRotaryEmbedding
from transformers.utils import logging
from transformers.processing_utils import Unpack
from datasets import load_dataset
import torch
import matplotlib.pyplot as plt
import os, sys
from typing import Optional, Tuple, Union, Callable
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from triton.language import dtype
# MODEL_ID="OPT"                 # NOTE: change the model_id here
HF_TOKEN="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"
DATASET = {                                                 # NOTE: change the dataset config here
    "name":"joey234/mmlu-high_school_biology",
    "config_name": "default",
    "split": "test",
    "input_feature_1": "question",
    "input_feature_2": "choices",
    "output_feature": "answer",
    }
BATCH_SIZE=3
TEXT_LENGTH=13
DS_SPLIT = 'test'
logger = logging.get_logger(__name__)

def llamarmsnorm_forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    w = self.weight
    tt = (self.weight * hidden_states)[0]
    hidden_states_0 = hidden_states[0]

    return self.weight * hidden_states.to(input_dtype)


def llamamodel_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )


    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for i in range(len(self.layers[: self.config.num_hidden_layers])):
        decoder_layer = self.layers[: self.config.num_hidden_layers][i]
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    output = BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
    return output if return_dict else output.to_tuple()

def llamamlp_forward(self,x):
    gate_proj = self.gate_proj(x)
    gate_act = self.act_fn(gate_proj)
    up_proj = self.up_proj(x)
    gate_act_up = gate_act * up_proj
    down_proj = self.down_proj(gate_act_up)
    return down_proj

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    print("cos: \t\t {}".format(cos.size()))
    print("sin: \t\t {}".format(sin.size()))
    print("q: \t\t {}".format(q.size()))
    print("k: \t\t {}".format(k.size()))

    print("----------\n(q * cos): \t\t {}".format((q * cos).size()))
    print("rotate_half(q): \t\t {}".format((rotate_half(q)).size()))
    print("(rotate_half(q) * sin): \t\t {}".format((rotate_half(q) * sin).size()))

    print("----------\n(k * cos): \t\t {}".format((k * cos).size()))
    print("rotate_half(k): \t\t {}".format((rotate_half(k)).size()))
    print("(rotate_half(k) * sin): \t\t {}\n".format((rotate_half(k) * sin).size()))

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    print("q_embed: \t\t {}".format(q_embed.size()))
    print("k_embed: \t\t {}".format(k_embed.size()))
    return q_embed, k_embed

def llamaattention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    print(f"head_dim\t\t{self.head_dim}")
    print(f"num_key_value_groups\t\t{self.num_key_value_groups}")
    print(f"num_key_value_heads\t\t{self.config.num_key_value_heads}")
    print(f"num_attention_heads\t\t{self.config.num_attention_heads}")

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    print("past_key_value:\t{}".format(past_key_value))
    print("self.config._attn_implementation:\t{}".format(self.config._attn_implementation))

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights



def llamarotaryembedding_forward(
    self, x, position_ids

):
    # Core RoPE block
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
    device_type = x.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

    # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
    cos = cos * self.attention_scaling
    sin = sin * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)




LlamaModel.forward = llamamodel_forward
LlamaRMSNorm.forward = llamarmsnorm_forward
LlamaMLP.forward = llamamlp_forward
LlamaAttention.forward = llamaattention_forward
LlamaRotaryEmbedding.forward = llamarotaryembedding_forward


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        use_auth_token=HF_TOKEN,
        torch_dtype=torch.float16
    )

    sum=0

    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Parameters: {param.numel()} | Element_Size: {param.element_size()}")
        sum = sum + param.numel()*param.element_size()

    print(model.get_memory_footprint(), sum)
    buffer = model.named_buffers()
    print(model.named_buffers())

    for name, buffer in model.named_buffers():
        print(f"Buffer: {name} | Size: {buffer.size()} | Parameters: {buffer.numel()} | Element_Size: {buffer.element_size()}")
        sum = sum + buffer.numel()*buffer.element_size()

    print(sum)

    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    return tokenizer, model

def lm_forward(input_shape=[BATCH_SIZE,TEXT_LENGTH], model_t=None, tokenizer=None):
    #encoded = tokenizer.encode(text, return_tensors="pt")
    input_tensor = torch.ones(*input_shape,dtype=torch.int)
    x = model_t(input_tensor.to(model_t.device)) # TODO: from here! Mar 24
    print(x.size())

def test_res():
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src import Result
    res = Result(
        model_id = None,
        model_layers = 1,
    )
def test_flops():
    from src import Flops
    flops = Flops(model_id=MODEL_ID)
    flops.get_config()

def test_torch_matmul():
    tensor1 = torch.randn(3, 4)
    tensor2 = torch.randn(4)
    a = torch.matmul(tensor1, tensor2).size()