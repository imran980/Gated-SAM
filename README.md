# GR3-SAM: Gated Recursive Refinement for Robust SAM

Official implementation of "Self-Correcting SAM: Robust Medical Image Segmentation via Gated Recursive Refinement"

## Key Results

| Dataset | SAM | MedSAM | Ours |
|---------|-----|--------|------|
| JSRT (noise=30) | 0.848 | 0.681 | **0.906** |
| BUSI (noise=30) | 0.647 | 0.565 | **0.673** |
| Kvasir (noise=30) | 0.670 | 0.462 | **0.724** |
| PROMISE12 (noise=30) | 0.443 | 0.311 | **0.497** |

## Installation
```bash
git clone https://github.com/imran980/Gated-SAM.git
cd Gated-SAM
pip install -r requirements.txt
```

## Quick Start
```python
from src.gated_refinement import GatedRefinementPredictor

predictor = GatedRefinementPredictor("sam_vit_b_01ec64.pth")
mask, confidence = predictor.predict(image, noisy_box)
```

## Citation
```bibtex
@inproceedings{imran2026gatedsam,
  title={Self-Correcting SAM: Robust Medical Image Segmentation via Gated Recursive Refinement},
  author={Muhammad Imran},
  booktitle={MICCAI},
  year={2026}
}
```
