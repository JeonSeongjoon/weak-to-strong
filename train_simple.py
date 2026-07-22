import json
import os
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import fire
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, load_from_disk, DatasetDict
from ruptures import Binseg
from transformers import TrainingArguments


import weak_to_strong.logger as logger
from weak_to_strong.common import get_tokenizer
from weak_to_strong.datasets import (VALID_DATASETS, load_dataset,
                                     tokenize_dataset)
from weak_to_strong.loss import (logconf_loss_fn, 
    product_loss_fn, 
    xent_loss, 
    conf_induc_loss
)
from weak_to_strong.train import train_and_save_model
from weak_to_strong.model import MODELS_DICT

from datacentric.sft import load_model_and_save_activations
from datacentric.sft_config import SFTConfig
from datacentric.probe import ProbeConfig, LogisticProbeConfig
from datacentric.model import ModelConfig as DLModelConfig
from datacentric.probe import PROBES



loss_dict = {
    "logconf": logconf_loss_fn(),
    "product": product_loss_fn(),
    "xent": xent_loss(),
    "conf_induc": conf_induc_loss()
}

VALID_LOSSES: List[str] = list(loss_dict.keys())


def get_config_foldername(config: dict) -> str:
    def shorten_key(key: str) -> str:
        return "".join(word[0] for word in key.split("_"))

    def shorten_value(value) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        elif isinstance(value, str):
            value = value.split("/")[-1]
            if "_" in value:
                return "_".join(word[:4] for word in value.split("_"))
            else:
                return value
        else:
            return str(value)

    return "-".join(f"{shorten_key(k)}={shorten_value(v)}" for k, v in sorted(config.items()))


def main(
    batch_size: int = 32,
    max_ctx: int = 1024,
    ds_name: str = "cosmos_qa",
    loss: str = "xent",
    n_docs: int = 20000,
    n_test_docs: int = 10000,
    model_size: str = "gpt2",
    lr: Optional[float] = None,
    optim: Optional[str] = None,
    epochs: int = 2,
    force_retrain: bool = False,
    seed: int = 0,
    minibatch_size_per_device: Optional[float] = None,
    train_with_dropout: bool = False,
    results_folder: str = "./results",
    linear_probe: bool = False,
    lr_schedule: str = "cosine_anneal",
    # Note: you can pass either weak_model_size or weak_labels_path. If you pass
    # weak_model_size, we will guess the path to the weak labels based on the weak
    # model. If you pass weak_labels_path, we will use that path instead.
    # If you pass neither, we will train on ground truth.
    weak_model_size: Optional[str] = None,
    weak_labels_path: Optional[str] = None,
    sweep_subfolder: str = "default",
    # Set to a very large value so that by default we don't do any intermediate evals but
    # still do final evals (which requires eval_every to be set to a non-zero, non-None value)
    eval_every: int = 1000000,
    sync_command: Optional[str] = None,
):

    # this is per device!
    if minibatch_size_per_device is None:
        minibatch_size_per_device = 1
    assert ds_name in VALID_DATASETS, f"Unknown dataset {ds_name} not in {VALID_DATASETS}"
    assert (
        weak_model_size is None or weak_labels_path is None
    ), "Can't pass both weak_model_size and weak_labels_path"
    model_config = MODELS_DICT[model_size]

    use_default_lr = False
    if lr is None:
        assert (
            batch_size == 32
        ), "Learning rates were tuned on batch size 32, you probably want to sweep LR if you are tuning batch size"
        lr = model_config.default_lr
        use_default_lr = True

    if optim is None:
        optim = model_config.default_optimizer

    shared_info_file_dir = f"./results/sample_difficulty/wms:{weak_model_size}_ms:{model_size}"

    # The commented out terms are the ones that should not change final results
    config = {
        "batch_size": batch_size,
        "max_ctx": max_ctx,
        "ds_name": ds_name,
        "loss": loss,
        "n_docs": n_docs,
        "n_test_docs": n_test_docs,
        "model_size": model_size,
        "lr": lr,
        "optim": optim,
        "epochs": epochs,
        # "force_retrain": force_retrain,
        "seed": seed,
        "minibatch_size_per_device": minibatch_size_per_device,
        "train_with_dropout": train_with_dropout,
        # "results_folder": results_folder,
        "linear_probe": linear_probe,
        "lr_schedule": lr_schedule,
        "eval_every": eval_every,
        # "sweep_subfolder": sweep_subfolder,
    }

    if weak_model_size is not None:
        weak_model_config = config.copy()
        weak_model_config["model_size"] = weak_model_size
        weak_model_config["loss"] = "xent"                  # it is revised from <loss> to <"xent">
        if use_default_lr:
            weak_model_config["lr"] = MODELS_DICT[weak_model_size].default_lr

        weak_model_config_name = get_config_foldername(weak_model_config)
        weak_labels_path = os.path.join(results_folder, sweep_subfolder, weak_model_config_name, "weak_labels")
           
    
    eval_batch_size = model_config.eval_batch_size
    random.seed(seed)

    # Load dataset
    dataset = load_dataset(ds_name, seed=seed, split_sizes=dict(train=n_docs, test=n_test_docs))

    # Split the training dataset in half
    train_dataset, test_ds = dataset["train"], dataset["test"]                 # train set + test set, validation set

    if weak_labels_path is None:
        split_data = train_dataset.train_test_split(test_size=0.5, seed=seed)       # train set : 10000
        train1_ds, train2_ds = split_data["train"], split_data["test"]              # validation set : 10000
        print("len(train1):", len(train1_ds), "len(train2):", len(train2_ds))       # test set : 10000
        config_name = get_config_foldername(config)
    else:
        if not weak_labels_path.endswith("weak_labels"):
            weak_labels_path = weak_labels_path + "/weak_labels"
            
        if sync_command is not None:
            sync_command_list = sync_command.split(" ")
            sync_command_list.extend(
                ["download", weak_labels_path.replace("/weak_labels", ""), results_folder]
            )
            print(f"Running sync command: {' '.join(sync_command_list)}")
            result = subprocess.run(sync_command_list, check=True)
            if result.returncode != 0:
                raise RuntimeError(f"Sync command failed with return code {result.returncode}")
        
        train1_ds = load_from_disk(weak_labels_path)
        print("Successfully load from disk.")
        train2_ds = None

        weak_model_config = json.load(open(weak_labels_path.replace("weak_labels", "config.json")))
        config["weak_model_size"] = weak_model_config["model_size"]
        config_name = get_config_foldername(config)
        config["weak_model"] = weak_model_config

    save_path = os.path.join(results_folder, sweep_subfolder, config_name)
    logger.configure(
        name="{sweep_subfolder}_{config_name}_{datetime_now}",
        save_path=save_path,
        sweep_subfolder=sweep_subfolder,
        config_name=config_name,
    )
    
    # Tokenize datasets
    tokenizer = get_tokenizer(model_config.name)
    train1_ds = tokenize_dataset(train1_ds, tokenizer, max_ctx)
    test_ds = tokenize_dataset(test_ds, tokenizer, max_ctx)
    if train2_ds:
        train2_ds = tokenize_dataset(train2_ds, tokenizer, max_ctx)
    train1_ds.save_to_disk(os.path.join(save_path, 'train_ds/')) 

    loss_fn = loss_dict[loss]

    
    # Train and evaluation
    print(f"Training model model, size {model_size}")
    test_results, weak_ds = train_and_save_model(
        model_config,
        train1_ds,
        test_ds,
        inference_ds=train2_ds,
        batch_size=batch_size,
        save_path=save_path,
        loss_fn=loss_fn,
        lr=lr,
        epochs=epochs,
        force_retrain=force_retrain,
        eval_batch_size=eval_batch_size,
        minibatch_size_per_device=minibatch_size_per_device,
        train_with_dropout=train_with_dropout,
        linear_probe=linear_probe,
        lr_schedule=lr_schedule,
        optimizer_name=optim,
        eval_every=eval_every,
        weak_model_size=weak_model_size,
        shared_info_file_dir=shared_info_file_dir
    )

    # Results
    if weak_ds is not None:
        save_path_wl = save_path + "/" + "weak_labels"
        weak_ds.save_to_disk(save_path_wl)
        test_results.save_to_disk(save_path_wl)

    test_results.save_to_disk(save_path)

    acc = np.mean([x["acc"] for x in test_results])
    res_dict = {"accuracy": acc}
    print("accuracy:", acc)

    with open(os.path.join(save_path, f"config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(save_path, f"results_summary.json"), "w") as f:
        json.dump(res_dict, f, indent=2)

    if sync_command is not None:
        print("Syncing results to remote storage...")
        try:
            sync_command_list = sync_command.split(" ")
            sync_command_list.extend(["upload", save_path, results_folder])
            print(f"Running sync command: {' '.join(sync_command_list)}")
            result = subprocess.run(sync_command_list, check=True)
            if result.returncode != 0:
                raise RuntimeError(f"Sync command failed with return code {result.returncode}")
        except Exception as e:
            raise RuntimeError("Failed to sync results to remote storage.") from e


    #############################################################
    #                        Activations                        #
    #############################################################

    shared_acts_dir = Path("./activations")
    
    # Caching activations 
    if weak_labels_path is None:
        cfg = SFTConfig(
            dataset=ds_name,
            n_train=len(train1_ds),             
            n_val=0,                                  
            n_test=len(test_ds),
            n_predict=0,
            minibatch_size=1,
            batch_size=32,
            results_folder="../../results",   # results_folder 수정 필요
            seed=seed,                        # seed값을 적절히 입력해야함.
            disable_lora=True,
            strong_only=True,
            probe=LogisticProbeConfig(),
            run_name=f"{ds_name}_{seed}",
        )

        train_args: dict = dict(
            num_train_epochs=cfg.n_epochs,
            adam_beta2=0.95,
            gradient_accumulation_steps=cfg.batch_size // cfg.minibatch_size,
            eval_strategy="steps",
            label_names=["labels"],
            load_best_model_at_end=cfg.load_best_model_at_end,
            logging_steps=25,
            metric_for_best_model=cfg.metric_for_best_model,
            greater_is_better=cfg.greater_is_better,
            per_device_train_batch_size=cfg.minibatch_size,
            per_device_eval_batch_size=cfg.minibatch_size,
            save_strategy="steps",
            save_total_limit=cfg.save_total_limit,
            tf32=True,  # Use Tensor Cores even for fp32 matmuls
            warmup_steps=cfg.n_warmup_steps,
            weight_decay=cfg.weight_decay,
            lr_scheduler_type=cfg.lr_schedule,
            eval_steps=cfg.eval_every,
        )

        def get_model_and_run_name(model_name, current_name):
            model_last = model_name.split("/")[-1]
            model_cfg = DLModelConfig(name=model_name, enable_lora=not cfg.disable_lora)
            run_name = f"{current_name}-{cfg.run_name}-{cfg.dataset}-{model_last}"
            return model_cfg, run_name

        shared_root = Path(cfg.results_folder) / "dcl"
        cfg_name = f"{cfg.run_name}_{cfg.weak_model_name.split('/')[-1]}_{cfg.strong_model_name.split('/')[-1]}"

        # train weak floor, get predictions
        print("\n\033[32m===== Training w2s model =====\033[0m")
        model_cfg, weak_run_name = get_model_and_run_name(cfg.weak_model_name, "weak")
        train_args["run_name"] = weak_run_name
        train_args["output_dir"] = str(shared_root / cfg_name / "output")
        train_args["learning_rate"] = cfg.weak_lr


        act_ds_dict = DatasetDict(
            {
                "train" : test_ds,                 # Already tokenized above
                "inference" : train2_ds,
            }
        )
        
        acts_dir = shared_acts_dir / f"ms:{model_size}"
        acts_dir.mkdir(parents=True, exist_ok=True)
        
        load_model_and_save_activations(
            ds_dict = act_ds_dict,
            model_cfg = model_cfg,
            train_args = TrainingArguments(**train_args),
            acts_dir = acts_dir
        )


    # Classifying samples as easy / overlap / hard
    if weak_model_size is not None:
        probe_name = "logreg"
        probe_cfg = LogisticProbeConfig()

        # load activations
        weak_acts_dir = shared_acts_dir / f"ms:{weak_model_size}"
        strong_acts_dir = shared_acts_dir / f"ms:{model_size}"

        x_weak_train = torch.load(weak_acts_dir / f"weak_train.pt", map_location="cuda")
        x_strong_train = torch.load(strong_acts_dir / f"strong_train.pt", map_location="cuda")
        x_w2s_train_for_pseudolabeling = torch.load(weak_acts_dir / f"strong_train.pt", map_location="cuda")
        y_weak_train = torch.tensor(test_ds["hard_label"], device="cuda")

        print(f"Weak acts shape: {x_weak_train.shape}")
        print(f"Strong acts shape: {x_strong_train.shape}")

        # probing
        weak_probe = PROBES[probe_name](probe_cfg)
        weak_probe.fit(x_weak_train, y_weak_train)
        y_w2s_train_for_pseudolabeling = weak_probe.predict(x_w2s_train_for_pseudolabeling)


        # detaching for not influencing the original values
        y_w2s_train_for_pseudolabeling = y_w2s_train_for_pseudolabeling.cpu().detach().numpy()
        y_w2s_train_for_pseudolabeling_indices = np.arange(
            len(y_w2s_train_for_pseudolabeling)
        ).reshape(-1,1)
        y_w2s_train_psdo_tb = np.hstack([
            y_w2s_train_for_pseudolabeling_indices,
            y_w2s_train_for_pseudolabeling.copy().reshape(-1,1)
            ]
        )
        
        x_strong_train = x_strong_train.cpu().detach().numpy()
        x_strong_train_indices = np.arange(len(x_strong_train)).reshape(-1,1)
        x_strong_train_tb = np.hstack([
            x_strong_train_indices, 
            x_strong_train.copy()
            ]
        )
        

        # Perform change point detection
        confidence_w2s_train = 2*np.abs(y_w2s_train_psdo_tb[:, 1] - 0.5)
        y_w2s_train_psdo_tb[:, 1] = confidence_w2s_train
        y_w2s_train_psdo_tb = y_w2s_train_psdo_tb[y_w2s_train_psdo_tb[:, 1].argsort()]  #np.sort(confidence_w2s_train)
        sorted_confidence = y_w2s_train_psdo_tb[:, 1].copy()
        
        model = Binseg(model="l2").fit(sorted_confidence.reshape(-1, 1))
        change_points = model.predict(n_bkps=1)[0]
        
        # Use the detected change point as the threshold
        confidence_threshold = sorted_confidence[change_points]
        low_confidence_indices = np.where(confidence_w2s_train <= confidence_threshold)[0]
        high_confidence_indices = np.where(confidence_w2s_train > confidence_threshold)[0]


        x_w2s_train_hard_tb = x_strong_train_tb[low_confidence_indices, :]              # (N x M)
        x_w2s_train_easy_or_overlap_tb = x_strong_train_tb[high_confidence_indices, :]  # (N x M)

        x_w2s_train_hard = x_w2s_train_hard_tb[:, 1:].copy()
        x_w2s_train_easy_or_overlap = x_w2s_train_easy_or_overlap_tb[:, 1:].copy()

        x_w2s_train_hard_normalized = x_w2s_train_hard / np.linalg.norm(x_w2s_train_hard, axis=1, keepdims=True)
        x_w2s_train_easy_or_overlap_normalized = x_w2s_train_easy_or_overlap / np.linalg.norm(x_w2s_train_easy_or_overlap, axis=1, keepdims=True)
        align_scores = np.abs(x_w2s_train_easy_or_overlap_normalized @ x_w2s_train_hard_normalized.T).max(axis=1)
        # align_scores = np.abs(x_w2s_train_easy_or_overlap @ x_w2s_train_hard.T).max(axis=1)

        # Apply change point detection to decide threshold for align scores
        sort_order = align_scores.argsort()
        x_w2s_train_easy_or_overlap_tb = x_w2s_train_easy_or_overlap_tb[sort_order, :]
        sorted_align_scores = align_scores[sort_order]
        # sorted_align_scores = np.sort(align_scores)
        
        # Perform change point detection
        model = Binseg(model="l2").fit(sorted_align_scores.reshape(-1, 1))
        change_points = model.predict(n_bkps=1)[0]
        
        # Use the detected change point as the threshold
        align_score_threshold = sorted_align_scores[change_points]
        overlap_indices = np.where(align_scores >= align_score_threshold)[0]
        nonoverlap_indices = np.where(align_scores < align_score_threshold)[0]

        x_w2s_train_overlap = x_w2s_train_easy_or_overlap_tb[overlap_indices, :]
        x_w2s_train_easy = x_w2s_train_easy_or_overlap_tb[nonoverlap_indices, :]
        
        # idx 별로 sort해서 pd.DataFrame으로 정리
        sample_indices = np.hstack([
            x_w2s_train_easy[:, 0],
            x_w2s_train_overlap[:, 0],
            x_w2s_train_hard_tb[:, 0],
            ]
        ).tolist()

        sample_diff_dict = {
            "idx" : sample_indices,
            "difficulty_label" : [0 for _ in range(len(x_w2s_train_easy))] 
            + [1 for _ in range(len(x_w2s_train_overlap))] 
            + [2 for _ in range(len(x_w2s_train_hard))]
        }

        pd.DataFrame(sample_diff_dict).to_csv(os.path.join(shared_info_file_dir, 'sample_info_label.csv'), index=False)
             


if __name__ == "__main__":
    fire.Fire(main)
