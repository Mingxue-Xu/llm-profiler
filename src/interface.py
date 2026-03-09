from typing import Dict, AnyStr, Optional, List
from .llama_hook import *
from .flops import INT64_ZERO
from transformers import AutoConfig, AutoModel

DISPLAY_STR="model:\t{}\n{} with input batchsize {} and sequence length {}:"
ATTR_STR = "    {}:\t{}"

def convert_number(num: INT64_ZERO) -> AnyStr:
    suffixes = ['', '\t(10^3)', '\t(10^6)', '\t(10^9)', '\t(10^12)', '\t(10^15)', '\t(10^18)']

    suffix_index = 0
    value = float(num)

    while abs(value) >= 1000 and suffix_index < len(suffixes) - 1:
        value /= 1000.0
        suffix_index += 1

    # Format with 1 decimal place if needed, otherwise as integer
    if value == int(value):
        return f"{int(value)}{suffixes[suffix_index]}"
    else:
        return f"{value:.2f}{suffixes[suffix_index]}"

def display(scope:AnyStr, kwargs: Dict, flopsunit: FlopsUnit, attr: AnyStr):
    attributes = vars(flopsunit)

    print(DISPLAY_STR.format(kwargs['model_id'],
                             scope,
                             kwargs['input_shape'][0],
                             kwargs['input_shape'][1])
          )
    if attr is None:
        for attr, value in attributes.items():
            print(ATTR_STR.format(attr.upper(),convert_number(value)))
    elif attr.lower() == "flops":
        print(ATTR_STR.format("FLOPSs (ADD and MULT)", convert_number(attributes['add'] + attributes['mult'])))
    elif attr.lower() in attributes.keys():
        print(ATTR_STR.format(attr.upper(), convert_number(attributes[attr.lower()])))
    else:
        raise NotImplementedError(f"{attr} is not supported")


def get_profile(
    kwargs:Dict=None,
    attr: Optional[AnyStr]=None,
    scope:Optional[AnyStr]=None,
):
    assert kwargs is not None, "get_profile(): `kwargs` cannot be None."
    assert 'model_config' in kwargs.keys(), "get_profile(): `kwargs.model_config` is not given."

    flopsunit = FlopsUnit()
    if scope is None or scope == "transformer":
        if scope is None:
            scope = "transformer"

        flopsunit=LlamaHook.llama_model(
            config=kwargs['model_config'],
            input_shape=kwargs['input_shape'],
            flopsunit=flopsunit,
        )

    elif scope == "attn":
        flopsunit = LlamaHook.llama_attention(
            config=kwargs['model_config'],
            input_shape=kwargs['input_shape'],
            flopsunit=flopsunit,
        )

    elif scope == "mlp":
        flopsunit = LlamaHook.llama_mlp(
            config=kwargs['model_config'],
            input_shape=kwargs['input_shape'],
            flopsunit=flopsunit,
        )

    display(scope=scope, kwargs=kwargs, flopsunit=flopsunit, attr=attr)






