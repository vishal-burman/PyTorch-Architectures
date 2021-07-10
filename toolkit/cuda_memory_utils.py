import gc
import torch


def is_cuda_out_of_memory(exception):
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "CUDA" in exception.args[0]
        and "out of memory" in exception.args[0]
    )


def is_cudnn_snafu(exception):
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]
    )


def is_out_of_cpu_memory(exception):
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "DefaultCPUAllocator: can't allocate memory" in exception.args[0]
    )


def is_oom_error(exception):
    return (
        is_cuda_out_of_memory(exception)
        or is_cudnn_snafu(exception)
        or is_out_of_cpu_memory(exception)
    )


def gc_cuda():
    gc.collect()
    if torch.cuda.is_available():
        try:  # Last thing which should cause OOM error but seemingly it can
            torch.cuda.empty_cache()
        except:
            if not is_oom_error(exception):
                raise
