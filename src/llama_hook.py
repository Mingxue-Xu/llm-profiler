import math
import torch
from typing import Optional, Tuple, List, Union, Dict
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.models.llama.modeling_llama import KwargsForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_rope_utils import _compute_default_rope_parameters, _compute_llama3_parameters
from .flops import *

class LlamaHookFunction(HookFunction):
    @staticmethod
    def llama_rms_norm(input_shape: List, config: LlamaConfig, flopsunit:Optional[FlopsUnit] = None):
        """
            https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L76
        """
        if flopsunit is None:
            flopsunit = FlopsUnit()

        batch_size, text_len, dim = input_shape[0], input_shape[1], config.hidden_size

        # variance = hidden_states.pow(2).mean(-1, keepdim=True)
        flopsunit.mult += batch_size * text_len * dim
        flopsunit.add += batch_size * text_len * (dim - 1)
        flopsunit.div += batch_size * text_len * 1
        flopsunit.cache += batch_size * text_len  # variance

        # hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        flopsunit.add += batch_size * text_len * 1
        flopsunit.div += batch_size * text_len * 1
        flopsunit.mult += batch_size * text_len * dim

        # self.weight * hidden_states (hidden_size)*(batch_size, text_len, dim) -> (batch_size, text_len, hidden_size), where dim = hidden_size

        flopsunit.params += dim + 1
        flopsunit.mult += batch_size * text_len * dim
        return flopsunit

    @staticmethod
    def llama_rotary_embedding_init(config: LlamaConfig, flopsunit: Optional[FlopsUnit] = None):
        if flopsunit is None:
            flopsunit = FlopsUnit()

        #   _compute_default_rope_parameters
        flopsunit.div += 1

        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)

        flopsunit.mult += 1
        flopsunit.div += dim
        flopsunit.mult += int(dim/2)

        #   _compute_llama3_parameters
        inv_freq, attention_factor = _compute_default_rope_parameters(config)
        factor = config.rope_scaling["factor"]  # `8` in the original implementation
        low_freq_factor = config.rope_scaling["low_freq_factor"]  # `1` in the original implementation
        high_freq_factor = config.rope_scaling["high_freq_factor"]  # `4` in the original implementation
        old_context_len = config.rope_scaling["original_max_position_embeddings"]  # `8192` in the original implementation
        low_freq_wavelen = old_context_len / low_freq_factor
        wavelen = 2 * math.pi / inv_freq

        flopsunit.div += 3
        flopsunit.mult += 1
        ###     inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        flopsunit.div += torch.sum(wavelen > low_freq_wavelen).item()
        ###     smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        flopsunit.add += 2
        flopsunit.div += 2
        ###     smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        flopsunit.add += 2
        flopsunit.div += 1
        flopsunit.mult += 2

        #   self.register_buffer("inv_freq", inv_freq, persistent=False)
        flopsunit.cache += config.head_dim

        #   self.original_inv_freq = self.inv_freq
        flopsunit.params +=config.head_dim

        return flopsunit

    @staticmethod
    def llama_rotary_embedding_forward(input_shape: List, config: LlamaConfig, flopsunit: Optional[FlopsUnit] = None):
        '''
            https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L128
        '''
        if flopsunit is None:
            flopsunit = FlopsUnit()

        batch_size, text_len, dim = input_shape[0], input_shape[1], config.hidden_size
        rope_factor = int(config.rope_scaling["factor"])
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            rope_type = "default"

        if "dynamic" in rope_type:
            raise NotImplementedError
        else:
            #   freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(1, 2)
            flopsunit.mult += batch_size * text_len * rope_factor
            #   emb = torch.cat((freqs, freqs), dim=-1)
            flopsunit.cache += text_len * config.head_dim

            #   cos = cos * self.attention_scaling
            flopsunit.mult += batch_size * text_len * rope_factor * 2
            #   sin = sin * self.attention_scaling
            flopsunit.mult += batch_size * text_len * rope_factor * 2

        #   return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        flopsunit.act += 2 * text_len * config.head_dim
        return flopsunit

    @staticmethod
    def llama_rotary_embedding(input_shape: List, config: LlamaConfig, flopsunit: Optional[FlopsUnit] = None):
        if flopsunit is None:
            flopsunit = FlopsUnit()
        flopsunit = LlamaHookFunction.llama_rotary_embedding_init(config=config, flopsunit=flopsunit)
        flopsunit = LlamaHookFunction.llama_rotary_embedding_forward(input_shape=input_shape, config=config, flopsunit=flopsunit)
        return flopsunit


    @staticmethod
    def llama_mlp(input_shape: List, config: LlamaConfig, flopsunit: Optional[FlopsUnit] = None):
        '''
            https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L196
        '''
        if flopsunit is None:
            flopsunit = FlopsUnit()

        batch_size, text_len, dim = input_shape[0], input_shape[1], config.hidden_size

        #   self.gate_proj(x)
        flopsunit = HookFunction.linear(batch_size=batch_size, in_features=config.hidden_size,
                                        out_features=config.intermediate_size, bias=config.mlp_bias, flopsunit=flopsunit)
        #   self.up_proj(x)
        flopsunit = HookFunction.linear(batch_size=batch_size, in_features=config.hidden_size,
                                             out_features=config.intermediate_size, bias=config.mlp_bias, flopsunit=flopsunit)
        #   self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        flopsunit.mult += config.hidden_size*config.intermediate_size
        #   self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        flopsunit = HookFunction.linear(batch_size=batch_size, in_features=config.intermediate_size,
                                        out_features=config.hidden_size, bias=config.mlp_bias,
                                        flopsunit=flopsunit)

        return flopsunit



    @staticmethod
    def llama_attention(
            input_shape: List,
            config: LlamaConfig = None,
            flopsunit: Optional[FlopsUnit] = None,
            ** kwargs:  Unpack[FlashAttentionKwargs],
    ):
        if flopsunit is None:
            flopsunit = FlopsUnit()

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        batch_size = input_shape[0]
        text_len = input_shape[1]

        # query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        flopsunit = HookFunction.linear(batch_size=batch_size, in_features=config.hidden_size,
                                                    out_features=config.num_attention_heads * head_dim, bias=config.attention_bias,
                                                    flopsunit=flopsunit)

        # key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        flopsunit = HookFunction.linear(batch_size=batch_size, in_features=config.hidden_size,
                                                    out_features=config.num_key_value_heads * head_dim, bias=config.attention_bias,
                                                    flopsunit=flopsunit)

        # value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        flopsunit = HookFunction.linear(batch_size=batch_size, in_features=config.hidden_size,
                                        out_features=config.num_key_value_heads * head_dim, bias=config.attention_bias,
                                        flopsunit=flopsunit)

        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        ### q_embed = (q * cos) + (rotate_half(q) * sin)
        ##### 1.    rotate_half(q), DIV +2
        flopsunit.div += 2
        ##### 2.    (rotate_half(q) * sin)
        flopsunit.mult +=   batch_size*config.num_attention_heads   *   text_len    *   head_dim
        ##### 3.    (q * cos)
        flopsunit.mult +=   batch_size*config.num_attention_heads   *   text_len    *  head_dim
        ##### 4.    (q * cos) + (rotate_half(q) * sin)
        flopsunit.add +=    batch_size*config.num_attention_heads   *   text_len    *  head_dim
        ### k_embed = (k * cos) + (rotate_half(k) * sin)
        ##### 1.    rotate_half(k), DIV +2
        flopsunit.div += 2
        ##### 2.    (rotate_half(k) * sin)
        flopsunit.mult +=   batch_size*config.num_key_value_heads   *   text_len    *   head_dim
        ##### 3.    (k * cos)
        flopsunit.mult +=   batch_size*config.num_key_value_heads   *   text_len    *  head_dim
        ##### 4.    (k * cos) + (rotate_half(k) * sin)
        flopsunit.add +=    batch_size*config.num_key_value_heads   *   text_len    *  head_dim

        # if past_key_value is not None:
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # # # key_states.size() = [batch_size,config.num_key_value_heads,text_len,head_dim]
        flopsunit.cache += batch_size*config.num_key_value_heads * text_len    *head_dim
        # # # value_states.size() = [batch_size,config.num_key_value_heads,text_len,head_dim]
        flopsunit.cache += batch_size * config.num_key_value_heads * text_len * head_dim

        if config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            raise NotImplementedError
        else:
            #   sdpa_attention_forward
            #   https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/sdpa_attention.py#L18
            #   L, S = query.size(-2), key.size(-2) ->  L=S=text_len
            #   scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
            #   attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
            flopsunit.div += 1
            flopsunit.sqrt += 1

            #   query @ key.transpose(-2, -1)
            flopsunit.mult += batch_size*config.num_attention_heads *   text_len    *   head_dim    *  text_len
            flopsunit.add +=  batch_size*config.num_attention_heads *   text_len    *   (head_dim-1)    *  text_len
            #   attn_weight = query @ key.transpose(-2, -1) * scale_factor
            flopsunit.mult += batch_size * config.num_attention_heads * text_len * text_len
            #   attn_weight += attn_bias
            flopsunit.add += batch_size * config.num_attention_heads * text_len * text_len
            #   attn_weight = torch.softmax(attn_weight, dim=-1)
            flopsunit.add +=    batch_size * config.num_attention_heads * text_len * (text_len-1)
            flopsunit.div +=    batch_size * config.num_attention_heads * text_len * text_len   #   TODO: is it correct here?
            #   attn_weight @ value
            flopsunit.mult += batch_size * config.num_attention_heads * text_len * head_dim * text_len
            flopsunit.add += batch_size * config.num_attention_heads * text_len * head_dim * (text_len-1)

        #   attn_output = self.o_proj(attn_output)
        flopsunit = HookFunction.linear(batch_size=batch_size, in_features=config.num_attention_heads * head_dim,
                                        out_features=config.hidden_size, bias=config.attention_bias,
                                        flopsunit=flopsunit)
        return flopsunit

class LlamaHook:
    @staticmethod
    def llama_decoder_layer(
        config: LlamaConfig,
        input_shape: List,
        flopsunit: Optional[FlopsUnit] = None,
    ):
        if flopsunit is None:
            flopsunit = FlopsUnit()
        batch_size, text_len, dim = input_shape[0], input_shape[1], config.hidden_size

        #   hidden_states = self.input_layernorm(hidden_states)
        flopsunit = LlamaHookFunction.llama_rms_norm(input_shape=input_shape, config=config, flopsunit=flopsunit)

        #   hidden_states, self_attn_weights = self.self_attn(...
        flopsunit = LlamaHookFunction.llama_attention(input_shape=input_shape, config=config, flopsunit=flopsunit)

        #   hidden_states = residual + hidden_states
        flopsunit.add += batch_size* text_len* dim

        #   hidden_states = self.post_attention_layernorm(hidden_states)
        flopsunit = LlamaHookFunction.llama_rms_norm(input_shape=input_shape, config=config, flopsunit=flopsunit)

        #   hidden_states = self.mlp(hidden_states)
        flopsunit = LlamaHookFunction.llama_mlp(input_shape=input_shape, config=config, flopsunit=flopsunit)

        #   hidden_states = residual + hidden_states
        flopsunit.add += batch_size * text_len * dim

        #   return outputs
        # flopsunit.act += batch_size* text_len* dim

        return flopsunit

    @staticmethod
    def llama_model(
        config: LlamaConfig,
        input_shape: List,
        flopsunit: Optional[FlopsUnit] = None,
    ):
        if flopsunit is None:
            flopsunit = FlopsUnit()

        batch_size, text_len, dim = input_shape[0], input_shape[1], config.hidden_size

        #   inputs_embeds = self.embed_tokens(input_ids)
        flopsunit = HookFunction.emb(batch_size=input_shape[0],text_len=input_shape[1],
                                     num_embeddings=config.vocab_size,
                                     embedding_dim=config.hidden_size,
                                     flopsunit=flopsunit)

        #   cache_position = torch.arange(...
        flopsunit.cache += text_len

        #   hidden_states = inputs_embeds
        flopsunit.cache += batch_size * text_len * dim

        #   position_embeddings = self.rotary_emb(hidden_states, position_ids)
        flopsunit = LlamaHookFunction.llama_rotary_embedding(input_shape=input_shape,config=config,flopsunit=flopsunit)


        #   for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        for i in range(config.num_hidden_layers):
        #       layer_outputs = decoder_layer(
            flopsunit =LlamaHook.llama_decoder_layer(
                    config=config,input_shape=input_shape,flopsunit=flopsunit)

        #   hidden_states = self.norm(hidden_states)
        flopsunit = LlamaHookFunction.llama_rms_norm(input_shape=input_shape, config=config, flopsunit=flopsunit)

        #   return output if return_dict else output.to_tuple() -> last_hidden_state and past_key_values

        return flopsunit

    @staticmethod
    def llama_attention(
        config: LlamaConfig,
        input_shape: List,
        flopsunit: Optional[FlopsUnit] = None,
    ):
        if flopsunit is None:
            flopsunit = FlopsUnit()

        for i in range(config.num_hidden_layers):
            flopsunit = LlamaHookFunction.llama_attention(
                config=config,input_shape=input_shape,flopsunit=flopsunit
            )
        return flopsunit

    @staticmethod
    def llama_mlp(
        config: LlamaConfig,
        input_shape: List,
        flopsunit: Optional[FlopsUnit] = None,
    ):
        if flopsunit is None:
            flopsunit = FlopsUnit()

        for i in range(config.num_hidden_layers):
            flopsunit = LlamaHookFunction.llama_mlp(
                config=config,input_shape=input_shape,flopsunit=flopsunit
            )
        return flopsunit






