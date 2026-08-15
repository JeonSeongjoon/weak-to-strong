import datasets
import numpy as np
import pandas as pd
import json
import torch

from pathlib import Path
from torch import nn



def to_batch(x, batch_size):
    for i in range(0, len(x), batch_size):
        yield x[i : i + batch_size]


def unpack(x):
    assert isinstance(x, torch.Tensor), type(x)
    return x.detach().float().cpu().numpy().tolist()


def eval_model_acc(model: nn.Module, ds: datasets.Dataset, eval_batch_size: int = 16) -> None:
    """
    This function evaluates the accuracy of a given model on a given dataset.

    Parameters:
    model (nn.Module): The model to be evaluated.
    ds (datasets.Dataset): The dataset on which the model is to be evaluated.

    Returns:
    results (list): A list of dictionaries containing the input_ids, ground truth label, predicted label,
                    accuracy of prediction, logits and soft label for each example in the dataset.
    """

    model.eval()

    with torch.no_grad():
        results = []
        gold_loss = 0
        weak_loss = 0
        n_total = 0

        # for ex in ds:
        for batch in to_batch(ds, eval_batch_size):

            # pad input_ids to common length
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ex) for ex in batch["input_ids"]], batch_first=True
            ).to(model.device if hasattr(model, "device") else "cpu")

            gt_labels = batch["gt_label"] if "gt_label" in batch else np.argmax(batch["soft_label"], axis=-1)
            sf_labels = batch["soft_label"]

            idxs = batch["idx"]
            # run forward pass
            raw_logits = model(input_ids)

            probs = unpack(torch.nn.functional.softmax(raw_logits, dim=-1))
            logits = unpack(raw_logits)

            preds = np.argmax(probs, axis=-1)
            labels = np.argmax(sf_labels, axis=-1)

            # calculate losses
            hard_t = torch.tensor(gt_labels, dtype=torch.long, device=raw_logits.device)  # for gt_labels
            soft_t = torch.tensor(np.asarray(sf_labels, dtype=np.float32), device=raw_logits.device)  # for weak_labels

            gold_loss += torch.nn.functional.cross_entropy(
                raw_logits.float(), hard_t, reduction="sum"
            ).item()

            weak_loss += torch.nn.functional.cross_entropy(
                raw_logits.float(), soft_t, reduction="sum"
            ).item()


            results.extend(
                [
                    dict(
                        idx=idx,
                        txt=txt,
                        input_ids=input_id,
                        gt_label=gt_label,
                        hard_label=pred,
                        acc=label == pred,
                        gt_acc=gt_label == pred,
                        logits=logit,
                        soft_label=prob,
                    )
                    for idx, input_id, txt, label, pred, prob, logit, gt_label in zip(
                        idxs, batch["input_ids"], batch["txt"], labels, preds, probs, logits, gt_labels
                    )
                ]
            )
            n_total += raw_logits.shape[0]

        accs = [r["acc"] for r in results]
        print("Accuracy:", np.mean(accs), "+/-", np.std(accs) / np.sqrt(len(accs)))

        return datasets.Dataset.from_list(results), gold_loss/n_total, weak_loss/n_total


def matchedness_eval(seed: int, loss: str):

    base_dir = Path(f"./weak-to-strong/results/sample_difficulty/seed={seed}")
    preds_root = base_dir / f"loss={loss}"
    labels_root = base_dir / "loss=xent"

    folders = [p.name for p in preds_root.iterdir() if p.is_dir()]

    for file_name in folders:
        file_parent_dir = preds_root / file_name

        # preds는 해당 loss 폴더에서, labels는 xent 폴더에서 읽는다
        sample_info_preds_dir = file_parent_dir / "sample_info.csv"
        sample_info_labels_dir = labels_root / file_name / "sample_info_label.csv"

        if not sample_info_preds_dir.exists() or not sample_info_labels_dir.exists():
            print(f"skip {file_name}: missing csv")
            continue

        sample_info_preds = pd.read_csv(sample_info_preds_dir)
        sample_info_labels = pd.read_csv(sample_info_labels_dir)

        sample_info_preds = sample_info_preds.drop_duplicates(subset="idx", keep="last")
        sample_info_labels = sample_info_labels.drop_duplicates(subset="idx", keep="last")

        # idx 기준으로 명시적으로 merge (정렬 후 위치로 비교하면 두 파일의 idx 집합이 어긋날 때 오정렬 위험이 있음)
        merged = pd.merge(
            sample_info_preds, 
            sample_info_labels[["idx", "difficulty_label"]], 
            on="idx", 
            how="inner"
        )
        merged.sort_values(
            "idx", 
            ignore_index=True, 
            inplace=True
        )

        label = merged["difficulty_label"].copy().map(
            lambda ex: 0 if ex == 0 or ex == 1 else 1
        )

        # Save the classification results
        results = merged["difficulty"] == label
        merged["matched"] = results
        merged.to_excel(file_parent_dir / "results_origin.xlsx")

        merged_cpy = merged.copy()
        merged_cpy.sort_values(
            ["matched", "difficulty_label", "difficulty"], 
            ascending = [False, True, False],
            ignore_index=True, 
            inplace=True
        )
        merged_cpy.to_excel(file_parent_dir / "results_sorted.xlsx")


        # Save the statistics
        matched_smp = merged[results]
        num_smp = len(merged)
        num_matched = len(matched_smp)
        num_unmatched = num_smp - num_matched
        matchedness_acc = num_matched / num_smp
        assert num_unmatched != 0, "The number of unmatched samples is 0. You cannot use it in division."

        num_matched_1 = len(merged[ (merged["difficulty"] == 0) & (merged["difficulty_label"] == 0)])
        num_matched_2 = len(merged[ (merged["difficulty"] == 0) & (merged["difficulty_label"] == 1)])
        num_matched_3 = len(merged[ (merged["difficulty"] == 1) & (merged["difficulty_label"] == 2)])
        matched_1_prop = num_matched_1 / num_matched; cor1_prop_total = num_matched_1 / num_smp
        matched_2_prop = num_matched_2 / num_matched; cor2_prop_total = num_matched_2 / num_smp
        matched_3_prop = num_matched_3 / num_matched; cor3_prop_total = num_matched_3 / num_smp

        num_unmatched_1 = len(merged[ (merged["difficulty"] == 0) & (merged["difficulty_label"] == 2)])
        num_unmatched_2 = len(merged[ (merged["difficulty"] == 1) & (merged["difficulty_label"] == 0)])
        num_unmatched_3 = len(merged[ (merged["difficulty"] == 1) & (merged["difficulty_label"] == 1)])
        unmatched_1_prop = num_unmatched_1 / num_unmatched; wrg1_prop_total = num_unmatched_1 / num_smp
        unmatched_2_prop = num_unmatched_2 / num_unmatched; wrg2_prop_total = num_unmatched_2 / num_smp
        unmatched_3_prop = num_unmatched_3 / num_unmatched; wrg3_prop_total = num_unmatched_3 / num_smp


        result_stats = {
            "#matched_samples": num_matched,
            "#unmatched_samples": num_unmatched,
            "matched_smaple_statistics":{
                "D:easy_or_ovlp-DL:easy_proportion": matched_1_prop,
                "D:easy_or_ovlp-DL:ovlp_proportion": matched_2_prop,
                "D:hard_DL:hard_proportion": matched_3_prop,
                "D:easy_or_ovlp-DL:easy_proportion(total)": cor1_prop_total,
                "D:easy_or_ovlp-DL:ovlp_proportion(total)": cor2_prop_total,
                "D:hard_DL:hard_proportion(total)": cor3_prop_total,
                "#D:easy_or_ovlp-DL:easy": num_matched_1,
                "#D:easy_or_ovlp-DL:ovlp": num_matched_2,
                "#D:hard_DL:hard": num_matched_3,
            },
            "unmatched_sample_statistics" : {
                "D:easy_or_ovlp-DL:hard_proportion": unmatched_1_prop,
                "D:hard_DL:easy_proportion": unmatched_2_prop,
                "D:hard_DL:ovlp_proportion": unmatched_3_prop,
                "D:easy_or_ovlp-DL:hard_proportion(total)": wrg1_prop_total,
                "D:hard_DL:easy_proportion(total)": wrg2_prop_total,
                "D:hard_DL:ovlp_proportion(total)": wrg3_prop_total,
                "#D:easy_or_ovlp-DL:hard": num_unmatched_1,
                "#D:hard-DL:easy": num_unmatched_2,
                "#D:hard-DL:ovlp": num_unmatched_3,
            },
            "matchedness_accuracy": matchedness_acc
        }

        results_dir = file_parent_dir / "matchedness_accuracy.json"
        with open(results_dir, "w") as f:
            json.dump(result_stats, f, indent=2)