import numpy as np
from transformers import PretrainedConfig
from typing import Optional
SUPPORTED_MODEL_ARCH = ['llama']

INT64_ZERO = np.int64(0)

class FlopsUnit:
    """
    function-wise flops container
    """
    add = INT64_ZERO
    mult = INT64_ZERO
    div = INT64_ZERO
    sqrt = INT64_ZERO
    params = INT64_ZERO
    cache = INT64_ZERO
    act = INT64_ZERO
    def __init__(self, add=INT64_ZERO, mult=INT64_ZERO, div=INT64_ZERO, sqrt=INT64_ZERO, params=INT64_ZERO, cache=INT64_ZERO, act=INT64_ZERO):
        self.add = add
        self.mult = mult
        self.div = div
        self.sqrt = sqrt
        self.params = params
        self.cache = cache

class HookFunction:
    """
        https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3646C1-L3662C19

        def get_memory_footprint(self, return_buffers=True):

        Get the memory footprint of a model. This will return the memory footprint of the current model in bytes.
        Useful to benchmark the memory footprint of the current model and design some tests. Solution inspired from the
        PyTorch discussions: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2

        Arguments:
            return_buffers (`bool`, *optional*, defaults to `True`):
                Whether to return the size of the buffer tensors in the computation of the memory footprint. Buffers
                are tensors that do not require gradients and not registered as parameters. E.g. mean and std in batch
                norm layers. Please see: https://discuss.pytorch.org/t/what-pytorch-means-by-buffers/120266/2

        mem = sum([param.nelement() * param.element_size() for param in self.parameters()])
        if return_buffers:
            mem_bufs = sum([buf.nelement() * buf.element_size() for buf in self.buffers()])
            mem = mem + mem_bufs
        return mem
    """

    @staticmethod
    def linear(batch_size, in_features, out_features, bias=True, flopsunit: Optional[FlopsUnit] = None):
        r"""Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.
        reference: torch._C._nn.linear

        x:      batch_size \times in_features
        A^T:    in_features \times out_features
        b:      batch_size \times out_features
        """

        if flopsunit is None:
            flopsunit = FlopsUnit()

        flopsunit.mult +=  batch_size * in_features * out_features

        if bias:
            bias_add = batch_size * out_features
            bias_mem = out_features
        else:
            bias_add = 0
            bias_mem = 0

        flopsunit.add += batch_size*(in_features-1)*out_features + bias_add

        flopsunit.params += batch_size * in_features * out_features + bias_mem    #  param.numel()

        flopsunit.act += batch_size * in_features * out_features #

        return flopsunit

    @staticmethod
    def emb(batch_size,text_len,num_embeddings: int, embedding_dim: int, flopsunit: FlopsUnit=None):

        if flopsunit is None:
            flopsunit = FlopsUnit()

        flopsunit.params += num_embeddings * embedding_dim
        flopsunit.act += batch_size * text_len * embedding_dim

        return flopsunit

class Flops:
    def __init__(self, model_id):
        assert any(substring in model_id for substring in SUPPORTED_MODEL_ARCH), f"The given model {model_id} is not supported"
        self.model_id = model_id

    def get_config(self):
        try:
            self.config = PretrainedConfig.from_pretrained(self.model_id)
        except AttributeError:
            print(f"Cannot get the config file of the model {self.model_id}")
        print(f"Config file of the model {self.model_id} is:\n {self.config}")

class LlamaFlops(Flops):
    def __init__(self, model_id):
        super().__init__(model_id)
        assert "llama" in model_id.lower(), f"The given model {model_id} is not Llama Architecture"

