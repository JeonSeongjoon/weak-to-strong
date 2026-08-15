import os
import subprocess
import sys
import json
from pathlib import Path
from typing import List, Union
import pandas as pd
import fire

from weak_to_strong.eval import matchedness_eval

def main(model_sizes: Union[List[str], str], **kwargs):
    if isinstance(model_sizes, str):
        model_sizes = model_sizes.split(",")
    assert (
        "weak_model_size" not in kwargs
        and "model_size" not in kwargs
        and "weak_labels_path" not in kwargs
    ), "Need to use model_sizes when using sweep.py"

    # configuration
    w2s_loss = kwargs.pop("loss", "xent")
    mode = kwargs.pop("mode", "w2sg")
    is_w2sg = mode == "w2sg"
    seed = kwargs.get("seed", 0)

    execution_file_dir = "train_simple.py" if is_w2sg else "train_difficulty.py"
    basic_args = [sys.executable, os.path.join(os.path.dirname(__file__), execution_file_dir)]
    
    for key, value in kwargs.items():
        basic_args.extend([f"--{key}", str(value)])

    if is_w2sg:
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
        is_conf_induc = w2s_loss.startswith("conf_induc") and not ("anc" in w2s_loss)
        if is_conf_induc:
            matchedness_eval(seed=seed, loss=w2s_loss)
            print("Save the matchedness results\n")
    else:
        # implement train_difficulty.py
        print("Obtaining anchor points")
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

if __name__ == "__main__":
    fire.Fire(main)
    

#[ LAB ver. ] -> tmux attach -t train
# CUDA_VISIBLE_DEVICES=0,1,2,3 python weak-to-strong/sweep.py --model_sizes=gpt2-large,gpt2-xl,Qwen/Qwen-1_8B,Qwen/Qwen-7B --seed=20 --loss=conf_induc --mode=w2sg --results_folder=./weak-to-strong/results/train_results/cosq_large_xent_20 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python weak-to-strong/sweep.py --model_sizes=gpt2-large,gpt2-xl,Qwen/Qwen-1_8B,Qwen/Qwen-7B --seed=20 --loss=conf_induc_anc --mode=diff --results_folder=./weak-to-strong/results/train_results/cosq_large_xent_20 
# CUDA_VISIBLE_DEVICES=2,3 python weak-to-strong/sweep.py --model_sizes=gpt2-large,gpt2-xl,Qwen/Qwen-1_8B --seed=15 --loss=xent --mode=w2sg --results_folder=./weak-to-strong/results/train_results/cosq_large_xent_15 --epochs=1