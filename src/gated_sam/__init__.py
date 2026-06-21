"""gated_sam — consistency-driven prompt-space optimization for robust SAM segmentation.

WACV 2027 reframe of "Gated Recursive Refinement":
  - a reference-free objective Q (perturbation-consistency) that needs no ground truth,
  - a prompt-space search that optimizes Q over a candidate neighborhood,
  - a guarded return (best-Q mask over the whole trajectory) giving a no-regression guarantee.
"""

__version__ = "0.2.0"
