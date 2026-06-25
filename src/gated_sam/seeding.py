"""Deterministic, process-independent RNG seeding.

Python's built-in hash() is salted per process (PYTHONHASHSEED), so seeding an RNG with
hash(name) gives different prompts every run. These helpers derive a stable seed from the
string form of the parts, so the same (sample, noise, seed, ...) always yields the same
perturbation — a hard requirement for reproducible tables.
"""
from __future__ import annotations

import hashlib

import numpy as np


def stable_seed(*parts) -> int:
    key = "|".join(str(p) for p in parts)
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)


def stable_rng(*parts) -> np.random.Generator:
    return np.random.default_rng(stable_seed(*parts))
