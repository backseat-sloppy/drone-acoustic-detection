"""Ensembles on a finished dev_test.py run: the mean confidence of several models, hop by hop.

    uv run python scripts/dev_test_ensemble.py TAG "name=model a+model b" ...

Models with different windows give different hop counts per round; streams are causal, so the hops are
aligned at the end of each round and the longer stream is cut at its start. The ensemble is written as
<TAG>/<name>.npz, next to the models, so `dev_test.py judge TAG` reads it like any model.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __import__("pathlib").Path(__file__).parent.as_posix())
import dev_test as D  # noqa: E402


def main(tag: str, specs: list[str]) -> None:
    out = D.OUT / tag
    for spec in specs:
        name, members = spec.split("=", 1)
        runs = []
        for m in members.split("+"):
            z = np.load(out / f"{m}.npz")
            runs.append([z[k] for k in z.files])
        ens = []
        for parts in zip(*runs):
            n = min(len(p) for p in parts)
            ens.append(np.mean([p[len(p) - n:] for p in parts], axis=0))
        np.savez_compressed(out / f"{name}.npz", *ens)
        print(f"{name}: {len(runs)} models, {sum(map(len, ens))} hops")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
