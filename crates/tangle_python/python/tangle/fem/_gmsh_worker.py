"""A gmsh process for :func:`tangle.fem.tet_mesh`: meshes one fiber at a time.

``tet_mesh`` starts these itself as plain Python processes, not through
``multiprocessing``, so a script that calls it needs no ``__main__`` guard,
and a fiber that takes too long can be stopped by ending its process. Each
job arrives as a pickle on stdin and each answer leaves as a pickle on the
original stdout; anything gmsh prints goes to stderr.
"""

from __future__ import annotations

import os
import pickle
import signal
import sys
import threading
import time


def main(verbose: bool) -> None:
    answers = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # from here on, stdout (gmsh's included) is stderr
    sys.stdout = sys.stderr
    jobs = sys.stdin.buffer
    # Ctrl-C reaches tet_mesh too, which ends this process.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if os.name == "posix":
        threading.Thread(target=_end_with, args=(os.getppid(),), daemon=True).start()

    import gmsh

    from tangle.fem._tet import _mesh_fiber, _start_gmsh

    _start_gmsh(gmsh, verbose)
    try:
        while True:
            try:
                job = pickle.load(jobs)
            except EOFError:  # tet_mesh is done with this process
                return
            try:
                answer = ("mesh", _mesh_fiber(gmsh, *job))
            except Exception as error:  # tet_mesh reports it and stops the other processes
                answer = ("error", str(error))
            pickle.dump(answer, answers, protocol=pickle.HIGHEST_PROTOCOL)
            answers.flush()
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


def _end_with(parent: int) -> None:
    """End this process once ``parent`` has gone, even in the middle of a
    fiber, so a killed ``tet_mesh`` leaves nothing running."""
    while os.getppid() == parent:
        time.sleep(2.0)
    os._exit(1)
