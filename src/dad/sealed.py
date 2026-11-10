"""ITU ARIS is a test set and nothing else (CLAUDE.md section 5).

Every reader that feeds a training or tuning run passes what it is about to read through
`forbid_in_training`, which stops the run if any of it is ITU ARIS or derived from it. There is
deliberately no flag to switch this off. Scoring code does not call it.

The check reads names and paths, not audio: a renamed copy gets through, so the rule in CLAUDE.md
still applies to anything this cannot see.
"""
from __future__ import annotations

import re

# The negative lookarounds keep "aris" a word of its own, so "comparison" and "Paris" pass;
# "_" is not in the class, so ITU_ARIS and aris_peek are caught.
_ITU_ARIS = re.compile(
    r"(?<![a-z0-9])aris(?![a-z0-9])"
    r"|sealed[\W_]*peek|steady[\W_]*peek|cleandrone"
    r"|extracted[\\/]+all[\\/]",
    re.IGNORECASE,
)


# The sealed car of The Final Evaluation (FSD50K eval 278188, scripts/fetch_sealed_car.py).
_SEALED_CAR = re.compile(r"sealed[\W_]*car|(?<!\d)278188(?!\d)", re.IGNORECASE)


class SealedDataError(RuntimeError):
    """Something on its way into training is ITU ARIS."""


def forbid_in_training(*names: object, where: str) -> None:
    """Stop the run if any of `names` (paths, source names, label text) points at ITU ARIS."""
    for name in names:
        if _SEALED_CAR.search(str(name)):
            raise SealedDataError(
                f"{where}: {name!s:.200} is the sealed car, kept for The Final Evaluation; never for training or tuning."
            )
        if _ITU_ARIS.search(str(name)):
            raise SealedDataError(
                f"{where}: {name!s:.200} is ITU ARIS or derived from it. ITU ARIS is only for "
                "testing finished models, never for training or tuning (CLAUDE.md section 5)."
            )
