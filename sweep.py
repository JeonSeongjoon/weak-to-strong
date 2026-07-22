import os
import subprocess
import sys
import json
from pathlib import Path
from typing import List, Union
import pandas as pd

import fire


def main(model_sizes: Union[List[str], str], **kwargs):
    if isinstance(model_sizes, str):
        model_sizes = model_sizes.split(",")
    assert (
        "weak_model_size" not in kwargs
        and "model_size" not in kwargs
        and "weak_labels_path" not in kwargs
    ), "Need to use model_sizes when using sweep.py"
    basic_args = [sys.executable, os.path.join(os.path.dirname(__file__), "train_simple.py")]
    w2s_loss = kwargs.pop("loss", "xent")
    
    for key, value in kwargs.items():
        basic_args.extend([f"--{key}", str(value)])
    
    # STEP1
    print("Running ground truth models")
    for model_size in model_sizes:
      subprocess.run(basic_args + ["--model_size", model_size], check=True)

    # STPE2
    print("Running transfer models")
    for i in range(len(model_sizes)):
        for j in range(i, len(model_sizes)):
            weak_model_size = model_sizes[i]
            strong_model_size = model_sizes[j]
            print(f"Running weak {weak_model_size} to strong {strong_model_size}")
            subprocess.run(
                basic_args
                + ["--weak_model_size", weak_model_size, "--model_size", strong_model_size, "--loss", w2s_loss],
                check=True,
            )
            
    # STEP3 : Evaluate the matchedness (only easy, overlap, hard)
    shared_file_dir = Path("./results/sample_difficulty")
    folders = [p.name for p in shared_file_dir.iterdir() if p.is_dir()]

    for file_name in folders:
        file_parent_dir = shared_file_dir / file_name

        # Evaluate the matchedness between sample predictions and labels
        sample_info_preds_dir = file_parent_dir / "sample_info.csv"
        sample_info_labels_dir = file_parent_dir / "sample_info_label.csv"

        sample_info_preds = pd.read_csv(sample_info_preds_dir)
        sample_info_labels = pd.read_csv(sample_info_labels_dir)

        # epochs > 1이면 같은 idx가 에폭마다 반복 기록되므로, 각 idx의 마지막(최종 에폭) 기록만 남긴다
        sample_info_preds = sample_info_preds.drop_duplicates(subset="idx", keep="last")

        sample_info_labels["difficulty_label"] = sample_info_labels["difficulty_label"].map(
            lambda ex: 0 if ex == 0 or ex == 1 else 1
        )

        # idx 기준으로 명시적으로 merge (정렬 후 위치로 비교하면 두 파일의 idx 집합이 어긋날 때 오정렬 위험이 있음)
        merged = pd.merge(
            sample_info_preds, 
            sample_info_labels[["idx", "difficulty_label"]], 
            on="idx", 
            how="inner"
        )
        merged.sort_values("idx", ignore_index=True, inplace=True)

        # Save the results and accuracy
        results = merged["difficulty"] == merged["difficulty_label"]
        merged["correct"] = results

        correct_smp = merged[results]
        matchedness_acc = len(correct_smp) / len(merged)

        merged.to_excel(file_parent_dir / "results.xlsx")
        results_dir = file_parent_dir / "results.json"
        with open(results_dir, "w") as f:
            json.dump(matchedness_acc, f, indent=2)

        


if __name__ == "__main__":
    fire.Fire(main)
    

#[ LAB ver. ] -> tmux attach -t train
#CUDA_VISIBLE_DEVICES= python sweep.py --model_sizes=[gpt2-large,gpt2-xl,Qwen/Qwen-1_8B] --seed=0 --loss=conf_induc --results_folder=./cosq_mid_ci_0 --epochs=2 --ds_name=cosmos_qa
#CUDA_VISIBLE_DEVICES=0,1 python sweep.py --model_sizes=gpt2-large,gpt2-xl,Qwen/Qwen-1_8B --seed=0 --loss=conf_induc --results_folder=./train_results/cosq_mid_conf_induc_0  

