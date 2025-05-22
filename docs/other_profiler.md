# Torch Profiler
[`torch.autograd.profiler`](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/autograd/profiler.py) and 
It is very low-level (including events and functions), but only for pytorch operations.
But [`torch.autograd.profiler`](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/autograd/profiler.py) will be replaced by [`torch.profiler`(experimental)](https://github.com/pytorch/pytorch/blob/f3cf3ec591528e1fd4b56dba6da5a0d803c61dc9/torch/profiler/profiler.py) soon. [`torch.profiler`](https://pytorch.org/docs/main/profiler.html#intel-instrumentation-and-tracing-technology-apis) is integrated with Intel profilers!
