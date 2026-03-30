import numpy as _np

def bootstrap_fit(*args, **kwargs):
    """
    Legacy bootstrap fit; removed in favor of `scipy.stats.bootstrap` functionality.

    .. deprecated:: 3.1.0
        Use :func:`scipy.stats.bootstrap` instead or implement custom fit workflows.
    """
    raise NotImplementedError("bootstrap_fit has been removed. Use scipy.stats.bootstrap or custom fit workflows.")

# def lin_fit(*args, **kwargs):
#     return linear(*args, **kwargs)

# def model_fit(*args, **kwargs):
#     return curve(*args, **kwargs)