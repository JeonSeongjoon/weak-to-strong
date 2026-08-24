
import os
import json
import torch
import numpy as np
import pandas as pd
from ruptures import Binseg
from typing import List

from datacentric.sft import load_activations
from datacentric.probe import LogisticProbeConfig
from datacentric.probe import PROBES


def sample_dffidulty_classification(
    shared_acts_dir: str,
    shared_info_file_dir: str,
    result_dir: str,
    wms_4_file: str,
    ms_4_file: str,
    train_dataset,
    train1_ds,
    seed: int,
    loss: str,
):
    assert shared_acts_dir is not None, "shared activation directory is None"

    probe_name = "logreg"
    probe_cfg = LogisticProbeConfig()

    weak_acts_dir = shared_acts_dir / f"ms:{wms_4_file}"
    strong_acts_dir = shared_acts_dir / f"ms:{ms_4_file}"

    # load activations
    x_weak_train, weak_train_idx = load_activations(weak_acts_dir / "probe_train.pt")
    x_weak_w2s, weak_w2s_idx = load_activations(weak_acts_dir / "target.pt")
    x_strong_w2s, strong_w2s_idx = load_activations(strong_acts_dir / "target.pt")

    # idx
    gt_by_idx = {
        int(i): int(l) for i, l in zip(train_dataset["idx"], train_dataset["hard_label"])
    }

    fit_common = np.array(
        sorted(set(weak_train_idx.tolist()) & set(gt_by_idx)), dtype=np.int64
    )
    
    wf_pos = {int(v): i for i, v in enumerate(weak_train_idx)}
    X_fit = x_weak_train[[wf_pos[int(i)] for i in fit_common]]
    y_fit = torch.tensor([gt_by_idx[int(i)] for i in fit_common], device=X_fit.device)

    weak_probe = PROBES[probe_name](probe_cfg)
    weak_probe.fit(X_fit, y_fit)
    print(f"Probe fitted on {len(fit_common)} samples (GT half)")

    labeled_idx = set(int(i) for i in train1_ds["idx"])
    common = np.array(
        sorted(set(weak_w2s_idx.tolist()) & set(strong_w2s_idx.tolist()) & labeled_idx),
        dtype=np.int64,
    )
    print(
        f"weak tgt: {len(weak_w2s_idx)}, strong tgt: {len(strong_w2s_idx)}, "
        f"labeled ds: {len(labeled_idx)}, common: {len(common)}"
    )

    overlap_chk = set(fit_common.tolist()) & set(common.tolist())
    assert not overlap_chk, (
        f"probe fit set overlaps target set on {len(overlap_chk)} idx. "
        f"Stale activation cache?"
    )
    if len(common) < 10:
        raise RuntimeError(
            "Too few overlapping idx. Did you regenerate activations after the fix?"
        )

    wt_pos = {int(v): i for i, v in enumerate(weak_w2s_idx)}
    st_pos = {int(v): i for i, v in enumerate(strong_w2s_idx)}
    x_weak = x_weak_w2s[[wt_pos[int(i)] for i in common]]
    x_strong = x_strong_w2s[[st_pos[int(i)] for i in common]]

    p = weak_probe.predict(x_weak).detach().float().cpu().numpy().reshape(-1)

    # easy & ovlp / hard classification
    conf = 2 * np.abs(p - 0.5)
    sorted_conf = np.sort(conf)
    cp = Binseg(model="l2").fit(sorted_conf.reshape(-1, 1)).predict(n_bkps=1)[0]
    conf_thr = sorted_conf[min(cp, len(sorted_conf) - 1)]

    hard_mask = conf < conf_thr
    rest_mask = ~hard_mask
    if hard_mask.sum() == 0 or rest_mask.sum() == 0:
        raise RuntimeError(f"Degenerate confidence split: hard={hard_mask.sum()}")
    
    # easy / ovlp classification
    x_strong_np = x_strong.detach().float().cpu().numpy()
    Xh = x_strong_np[hard_mask]
    Xr = x_strong_np[rest_mask]
    Xh_n = Xh / np.linalg.norm(Xh, axis=1, keepdims=True)
    Xr_n = Xr / np.linalg.norm(Xr, axis=1, keepdims=True)
    align = np.abs(Xr_n @ Xh_n.T).max(axis=1)   
    # Find the most similar pair between weak, strong pairs

    sorted_align = np.sort(align)
    cp2 = Binseg(model="l2").fit(sorted_align.reshape(-1, 1)).predict(n_bkps=1)[0]
    align_thr = sorted_align[min(cp2, len(sorted_align) - 1)]

    print(f"align: mean={align.mean():.4f}, std={align.std():.4f}")
    print(f"percentiles: {np.percentile(align, [1, 25, 50, 75, 99])}")
    print(f"align_thr={align_thr:.4f}")

    hard_idx = common[hard_mask]
    rest_idx = common[rest_mask]
    overlap_idx = rest_idx[align >= align_thr]
    easy_idx = rest_idx[align < align_thr]

    # length of datasets
    num_easy = len(easy_idx)
    num_ovlp = len(overlap_idx)
    num_hard = len(hard_idx)
    num_total = num_easy + num_ovlp + num_hard

    print(f"easy={num_easy}, overlap={num_ovlp}, hard={num_hard}")

    sample_diff_dict = {
        "idx": np.concatenate([easy_idx, overlap_idx, hard_idx]).tolist(),
        "difficulty_label": [0] * num_easy
        + [1] * num_ovlp
        + [2] * num_hard,
    }

    # Save the classification results
    print("========== Save the classification results ==========\n")
    pd.DataFrame(sample_diff_dict).to_csv(
        os.path.join(shared_info_file_dir, "sample_info_label.csv"), index=False
    )

    # Save the classified dataset
    diff_ds_prnt_dir = result_dir + f"/diff_ds/seed={seed}/loss={loss}"
    pair_dir = diff_ds_prnt_dir + f"/wms:{wms_4_file}_ms:{ms_4_file}"
    os.makedirs(pair_dir, exist_ok=True)

    idx_to_pos = {int(v): i for i, v in enumerate(train1_ds["idx"])}

    for diff, idx_set in dict(
        easy=easy_idx, overlap=overlap_idx, hard=hard_idx
    ).items():
        missing = [int(i) for i in idx_set if int(i) not in idx_to_pos]
        assert not missing, f"{len(missing)} idx missing from train1_ds: {missing[:5]}"

        positions = [idx_to_pos[int(i)] for i in idx_set]
        ds = train1_ds.select(positions)
        ds = ds.add_column("difficulty", [1 if diff == "hard" else 0] * len(ds))

        ds.save_to_disk(os.path.join(pair_dir, f"{diff}_ds"))

    diff_ds_length = {
        "num_easy": num_easy,
        "num_overlap": num_ovlp,
        "num_hard": num_hard,
        "%_easy": num_easy / num_total,
        "%_overlap": num_ovlp / num_total,
        "%_hard": num_hard / num_total,
        "num_total": num_total
    }

    with open(os.path.join(pair_dir, "diff_ds_length.json"), "w") as f:
        json.dump(diff_ds_length, f, indent=2)