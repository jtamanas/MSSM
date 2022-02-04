import jax.numpy as np
import numpy as onp
from typing import Callable, List, Optional
import multiprocessing as mp
import tempfile
import os
import random


class Worker:
    def __init__(self, f, chunk, chdir=True) -> None:
        self.f = f
        self.chunk = chunk
        self.chdir = chdir

    def work(self, cdir):
        if self.chdir:
            with tempfile.TemporaryDirectory() as tdir:
                os.chdir(tdir)
                results = [self.f(args) for args in self.chunk]
                os.chdir(cdir)
            return results
        else:
            return [self.f(args) for args in self.chunk]


def work(worker, cdir):
    return worker.work(cdir)


def apply_distributed(
    f: Callable,
    args_array: List,
    nprocs: Optional[int] = None,
    chdir: Optional[bool] = True,
) -> List:
    """
    Apply the function over the array of arguments distributed over mutliple processes.
    Parameters
    ----------
    f: Callable
        Function to run.
    args_array: List
        List of arguments to apply function to. Function must be able to call using:
            f(*args_array[0])
    nprocs: int, Optional
    """
    if nprocs is None:
        nprocs = mp.cpu_count() - 1
        
    chunks = [args_array[i : i + nprocs] for i in range(0, len(args_array), nprocs)]
    workers = [Worker(f, chunk, chdir) for chunk in chunks]
    procs = mp.Pool(len(chunks))
    cdir = os.getcwd()
    results = []
    for worker in workers:
        results.append(procs.apply_async(work, (worker, cdir)))
    procs.close()
    procs.join()
    # import IPython; IPython.embed()
    out = []
    for result in results:
        outputs = result.get()
        for output in outputs:
            out.append(output[0])
    results = onp.array(out)
    results = results.reshape(len(args_array), -1)
    return results


def test_f(x, y):
    return [x * y, os.getcwd()]


if __name__ == "__main__":
    args_array = [(random.random(), random.random()) for _ in range(100)]
    results = apply_distributed(test_f, args_array)
    for res in results:
        for r in res:
            print(r)
        print()
