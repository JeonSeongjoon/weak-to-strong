import os
import subprocess
import sys
from typing import List, Union

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
    
    print("Running ground truth models")
    for model_size in model_sizes:
      subprocess.run(basic_args + ["--model_size", model_size], check=True)

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
            

if __name__ == "__main__":
    fire.Fire(main)
    
    
#python sweep.py --model_sizes=gpt2,gpt2-medium,gpt2-large --seed=0 --loss=conf_induc --results_folder=./result_conf_induc_0

#[ LAB ver. ] -> tmux attach -t train
#CUDA_VISIBLE_DEVICES= python sweep.py --model_sizes=[gpt2-large,gpt2-xl,Qwen/Qwen-1_8B] --seed=0 --loss=conf_induc --results_folder=./cosq_mid_ci_0 --epochs=2 --ds_name=cosmos_qa
#CUDA_VISIBLE_DEVICES=0,1 python sweep.py --model_sizes=gpt2-large,gpt2-xl,Qwen/Qwen-1_8B --seed=0 --loss=conf_induc --results_folder=./train_results/cosq_mid_conf_induc_0  

