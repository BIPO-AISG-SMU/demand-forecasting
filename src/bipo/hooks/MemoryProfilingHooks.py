import logging
from typing import Union, List, Tuple
from kedro.framework.hooks import hook_impl
from memory_profiler import memory_usage


def _normalise_mem_usage(mem_usage: Union[List, Tuple, float]):
    """This function returns the memory usage value
    Args:
        mem_usage (Union[List, Tuple, float]): Memory usage input as observed by memory_profiler

    Raises:
        None.

    Returns:
        flat: Memory usage in MiB
    """
    # memory_profiler < 0.56.0 returns list instead of float
    return mem_usage[0] if isinstance(mem_usage, (list, tuple)) else mem_usage


class MemoryProfilingHooks:
    """This class calculates the memory utilised when dataset are being loaded. This is copied from https://docs.kedro.org/en/stable/hooks/examples.html."""

    def __init__(self, LOGGER_NAME):
        self._mem_usage = {}
        self.logger = logging.getLogger(LOGGER_NAME)

    @hook_impl
    def before_dataset_loaded(self, dataset_name: str) -> None:
        """This function checks the current memory utilisation prior to Kedro dataset loading process and stores into a class dictionary attribute.

        Args:
            dataset_name (str): Kedro dataset name that is processed.
        """
        before_mem_usage = memory_usage(
            -1,
            interval=0.1,
            max_usage=True,
            retval=True,
            include_children=True,
        )
        before_mem_usage = _normalise_mem_usage(before_mem_usage)
        self._mem_usage[dataset_name] = before_mem_usage

    @hook_impl
    def after_dataset_loaded(self, dataset_name: str) -> None:
        """This function checks the current memory utilisation after Kedro dataset loading process and stores into a class dictionary attribute and logs the difference between before and after dataset loading.

        Args:
            dataset_name (str): Kedro dataset name that is processed.
        """
        after_mem_usage = memory_usage(
            -1,
            interval=0.1,
            max_usage=True,
            retval=True,
            include_children=True,
        )
        # memory_profiler < 0.56.0 returns list instead of float
        after_mem_usage = _normalise_mem_usage(after_mem_usage)

        logging.getLogger("kedro").info(
            "Loading %s consumed %2.2fMiB memory",
            dataset_name,
            after_mem_usage - self._mem_usage[dataset_name],
        )
