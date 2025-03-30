
# FLOPs calculation references
## Scaled Dot Product Attention
Python version (not actually executed) of `scaled_dot_product_attention`: [`torch.nn.functional.scaled_dot_product_attention`](https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py#L5720)

```python
# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

```

`.cpp` version of `scaled_dot_product_attention`: [`torch._C._nn.scaled_dot_product_attention`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/attention.cpp#L696)

```c

// Computes scaled dot product attention on query, key and value tensors, using
// an optional attention mask if passed, and applying dropout if a probability
// greater than 0.0 is specified.
//
// Args:
//     query (Tensor): Query tensor; shape (N, ..., L, E)
//     key (Tensor): Key tensor; shape (N, ..., S, E)
//     value (Tensor): Value tensor; shape (N, ..., S, E)
//     attn_mask (optional Tensor): Attention mask; shape must be broadcastable to the shape of attention weights,
//         which is (N,..., L, S). Two types of masks are supported.
//         A boolean mask where a value of True indicates that the element *should* take part in attention.
//         A float mask of the same type as query, key, value that is added to the attention score.
//     dropout_p (float): Dropout probability; if greater than 0.0, dropout is applied
//     need_attn_weights (bool): If true, the second return value will contain the attention weights used;
//         otherwise, the second return value is unspecified
//     is_causal (bool): If true, assumes causal attention masking; for this case, attn_mask should not be set.
//         TODO: Consider removing this flag before promoting this function to the public API. It's possible
//         to get specialized support for causal masks (and other types of masking e.g. local attention / block
//         sparse masks) via tensor subclassing, allowing for a leaner API.
//
// Returns a tensor:
//     output (Tensor): Attention output; shape (N, ..., L, E)
//
// Shape legend:
//     N: Batch size
//     ...: Any number of other batch dimensions (optional)
//     S: Source sequence length
//     L: Target sequence length
//     E: Embedding dimension

Tensor scaled_dot_product_attention(
    const Tensor& query_,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  using sdp::SDPBackend;
  validate_sdpa_input(query_, key, value, attn_mask_, dropout_p, is_causal, scale);
  int64_t choice_int = static_cast<int64_t>(sdp::SDPBackend::math);
  if (_fused_sdp_choice_stub.is_device_supported(query_.device().type())) {
    choice_int = _fused_sdp_choice_stub(query_.device().type(),
          query_, key, value, attn_mask_, dropout_p, is_causal, scale, enable_gqa);
  }
  const auto query_device_type = query_.device().type();
  const auto backend = static_cast<SDPBackend>(choice_int);
  const auto convert_attn_func = backend != SDPBackend::cudnn_attention ? convert_boolean_attn_mask : convert_boolean_attn_mask_cudnn;
  auto attn_mask = convert_attn_func(attn_mask_, query_.dtype());
  switch (backend) {
    case SDPBackend::cudnn_attention: {
      bool compute_logsumexp = should_compute_logsumexp(query_, key, value);
      auto out_lse_softmax = at::_scaled_dot_product_cudnn_attention(
          query_, key, value, attn_mask, compute_logsumexp, dropout_p, is_causal, false /*return_debug_mask*/, scale);
      return std::get<0>(out_lse_softmax);
    }
    case SDPBackend::flash_attention: {
      if(query_device_type == DeviceType::CUDA){
        c10::SymInt og_size = query_.sym_size(-1);
        Tensor query_padded = pad_last_dim<8, false>(query_);
        Tensor key_padded = pad_last_dim<8, false>(key);
        Tensor value_padded = pad_last_dim<8, false>(value);
        // We need to calculate the scale based off the OG head dim size
        auto og_scale = sdp::calculate_scale(query_, scale);
        auto out_lse_softmax = at::_scaled_dot_product_flash_attention(
            query_padded, key_padded, value_padded, dropout_p, is_causal, false /*return_debug_mask*/, og_scale.guard_float("attention.cpp", 735));
        return post_process_flash_output(std::get<0>(out_lse_softmax), og_size);
      }
      // For the CPU case we do not need to pad the last dim
      return std::get<0>(at::_scaled_dot_product_flash_attention_for_cpu(
          query_, key, value, dropout_p, is_causal, attn_mask, scale));
    }
    case SDPBackend::efficient_attention: {
      bool compute_logsumexp = should_compute_logsumexp(query_, key, value);
      if (attn_mask.has_value()) {
        attn_mask.value() = preprocess_mask(attn_mask.value(), query_, key, value);;
      }
      auto out_and_lse = at::_scaled_dot_product_efficient_attention(
          query_, key, value, attn_mask, compute_logsumexp, dropout_p, is_causal, scale);
      return std::get<0>(out_and_lse);
    }
    case SDPBackend::overrideable: {
      auto out_lse_softmax = at::_scaled_dot_product_fused_attention_overrideable(
          query_, key, value, attn_mask, dropout_p, is_causal, false /*return_debug_mask*/, scale);
      return std::get<0>(out_lse_softmax);
    }
    case SDPBackend::math: {
#ifdef USE_MPS
      const auto any_nested = query_.is_nested() || key.is_nested() || value.is_nested();
      const bool any_inputs_require_grad = query_.requires_grad() || key.requires_grad() || value.requires_grad();
      const auto all_contiguous = query_.is_contiguous() && key.is_contiguous() && value.is_contiguous();
      if (query_device_type == DeviceType::MPS && dropout_p == 0.0
          && !(GradMode::is_enabled() && any_inputs_require_grad)
          && (all_contiguous || mps::is_macos_13_or_newer(mps::MacOSVersion::MACOS_VER_15_0_PLUS))
          && !any_nested) {
        if (enable_gqa) {
          int64_t q_heads = query_.size(-3);
          int64_t k_heads = key.size(-3);
          int64_t repeat_factor = q_heads / k_heads;

          if (repeat_factor > 1) {
            TORCH_CHECK(q_heads % k_heads == 0,
                          "For GQA, the query tensor's head dimension (" + std::to_string(q_heads) +
                                    ") must be divisible by the key tensor's head dimension (" + std::to_string(k_heads) + ").");
            auto repeated_key = key.repeat_interleave(repeat_factor, /*dim=*/-3);
            auto repeated_value = value.repeat_interleave(repeat_factor, /*dim=*/-3);
            return std::get<0>(at::_scaled_dot_product_attention_math_for_mps(
              query_,
              repeated_key,
              repeated_value,
              attn_mask,
              dropout_p,
              is_causal,
              std::nullopt, /*dropout_mask*/
              scale));
          }
        }
        return std::get<0>(at::_scaled_dot_product_attention_math_for_mps(
            query_,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            std::nullopt, /*dropout_mask*/
            scale));
      }
#endif
      return std::get<0>(at::_scaled_dot_product_attention_math(
          query_,
          key,
          value,
          attn_mask,
          dropout_p,
          is_causal,
          std::nullopt, /*dropout_mask*/
          scale,
          enable_gqa));
    }
    default:
      TORCH_CHECK(
          false,
          "No viable backend for scaled_dot_product_attention was found.");
      return Tensor();
  }
}
```