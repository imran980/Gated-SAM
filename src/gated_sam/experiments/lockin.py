"""Day 6 — fixed-point / lock-in analysis.

Runs the search at high noise, logs Q(M_k) and step-movement IoU(M_k, M_{k-1}), and
classifies each trajectory as recovery vs lock-in. Lock-in is the flat-/low-Q regime
where plain iteration cannot escape; the guarded search either escapes it or refuses
to move (no regression). Reports the recovery rate (headline: PROMISE12) and contrasts
the search against plain ungated iteration.

    python -m gated_sam.experiments.lockin --config configs/default.yaml --set noise_levels=[30]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..baselines import ungated_cascade
from ..data import load_dataset
from ..metrics import dice
from ..objectives import build_objective
from ..prompts import add_box_noise
from ..refine import refine_search
from . import _common

RECOVER_MARGIN = 0.05    # final must beat init by this to count as "recovered"
BAD_INIT = 0.6           # an initial prediction below this is "in trouble"


def analyze(sam, samples, cfg, dataset, noise):
    obj = build_objective(cfg)
    rows, traces = [], []
    for s in tqdm(samples, desc=f"lockin:{dataset}"):
        sam.set_image(s.image)
        h, w = s.image.shape[:2]
        rng = np.random.default_rng((hash(s.name) ^ (noise << 8)) % (2**32))
        box = add_box_noise(s.gt_box, int(noise), h, w, rng)

        res = refine_search(sam, s.image, box, obj, cfg, rng)
        init_dice = dice(res.masks[0], s.gt_mask)
        final_dice = dice(res.mask, s.gt_mask)
        # plain iteration (no gate, no consistency) for contrast
        plain = dice(ungated_cascade(sam, s.image, box, rng, cfg), s.gt_mask)

        recovered = final_dice > init_dice + RECOVER_MARGIN
        rows.append(dict(dataset=dataset, sample=s.name, noise=noise,
                         init_dice=init_dice, final_dice=final_dice, plain_dice=plain,
                         chosen_step=res.chosen_step, n_steps=res.n_steps, moved=res.moved,
                         recovered=recovered, bad_init=init_dice < BAD_INIT))
        traces.append(dict(sample=s.name, recovered=recovered, init_dice=init_dice,
                           final_dice=final_dice,
                           steps=[dict(k=st.k, Q=st.Q, move=st.move) for st in res.trajectory]))
    return rows, traces


def report(df):
    lines = ["", "=" * 64, "LOCK-IN / RECOVERY ANALYSIS", "=" * 64]
    for d, g in df.groupby("dataset"):
        bad = g[g.bad_init]
        rate = bad.recovered.mean() if len(bad) else float("nan")
        esc = (bad.final_dice > bad.plain_dice + 1e-6).mean() if len(bad) else float("nan")
        lines.append(f"  {d:<10} bad-init cases={len(bad):>3}  recovery-rate={rate:.0%}  "
                     f"beats-plain-iteration={esc:.0%}  mean Δdice={g.final_dice.mean()-g.init_dice.mean():+.3f}")
    lines.append("=" * 64)
    return "\n".join(lines)


def main(argv=None):
    p = _common.base_parser(__doc__)
    p.add_argument("--noise", type=int, default=30, help="noise level to analyze")
    p.add_argument("--n-traces", type=int, default=6, help="trajectories to plot per class")
    args = p.parse_args(argv)
    cfg = _common.get_config(args)
    out = _common.out_dir(cfg, "lockin")
    sam = _common.build_sam(cfg)

    all_rows, all_traces = [], []
    for name in _common.dataset_names(cfg, args):
        samples = load_dataset(name, cfg)
        if not samples:
            continue
        rows, traces = analyze(sam, samples, cfg, name, args.noise)
        all_rows += rows
        all_traces += traces

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "lockin.csv", index=False)
    print(report(df))

    rec = [t for t in all_traces if t["recovered"]][: args.n_traces]
    lock = [t for t in all_traces if not t["recovered"] and t["init_dice"] < BAD_INIT][: args.n_traces]
    try:
        from ..figures import lockin_trajectories
        if rec or lock:
            lockin_trajectories(rec + lock, out)
        print(f"\nLock-in tables + figures written to {out}")
    except Exception as exc:
        print(f"[warn] figure generation skipped: {exc}")
    return df


if __name__ == "__main__":
    main()
