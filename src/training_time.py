"""Wall-clock training measurements, separate from research metrics."""
import importlib.metadata
import os
import platform
from time import perf_counter

from threadpoolctl import threadpool_info


def timed_call(function, *args, **kwargs):
    started = perf_counter()
    result = function(*args, **kwargs)
    return result, perf_counter() - started


def execution_environment():
    return dict(os=platform.platform(), machine=platform.machine(), processor=platform.processor(),
                logical_cpus=os.cpu_count(), python=platform.python_version(),
                threadpools=threadpool_info(), packages={name: importlib.metadata.version(name)
                    for name in ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'joblib', 'pytest')})
