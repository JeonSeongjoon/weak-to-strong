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
    conf_induc_loss,
    conf_induc_filt_loss
)
from weak_to_strong.train import ModelConfig, train_and_save_model


from datacentric.sft import load_model_and_save_activations, load_activations
from datacentric.sft_config import SFTConfig
from datacentric.probe import ProbeConfig, LogisticProbeConfig
from datacentric.model import ModelConfig as DLModelConfig
from datacentric.probe import PROBES


# NOTE learning rates are not particularly tuned, work somewhat reasonably at train batch size 32
MODEL_CONFIGS = [
    ModelConfig(
        name="gpt2",
        default_lr=5e-5,
        eval_batch_size=32,
    ),
    ModelConfig(
        name="gpt2-medium",
        default_lr=5e-5,
        eval_batch_size=32,
    ),
    ModelConfig(
        name="gpt2-large",
        default_lr=1e-5,
        eval_batch_size=32,
        model_parallel=(                     
            torch.cuda.device_count() > 1
        ),
    ),
    ModelConfig(
        name="gpt2-xl",
        default_lr=1e-5,
        eval_batch_size=32,
        gradient_checkpointing=True,
        # Should use model_parallel on V100s (note: ironically if you have a single V100 it should run,
        # but if you have multiple it won't run without model_parallel because of the overhead of data
        # parallel training).
        model_parallel=(
            #torch.cuda.get_device_properties(0).total_memory < 50e9 and 
            torch.cuda.device_count() > 1
        ),
    ),
    ModelConfig(
        name="Qwen/Qwen-1_8B",
        default_lr=1e-5,
        eval_batch_size=32,
        gradient_checkpointing=True,
        model_parallel=(
            #torch.cuda.get_device_properties(0).total_memory < 50e9 and
            torch.cuda.device_count() > 1
        ),
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
            "revision": "5fde88dff770a7d036847211f5d9d9705f0caa69",
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-7B",
        default_lr=1e-5,
        eval_batch_size=8,
        gradient_checkpointing=True,
        model_parallel=True,                  
        # I set the model_parallel flag false for Colab environment
        # If you run this code in another environment, you have to set it True
        # note: you will probably not be able to run this without many gpus
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
            "revision": "d4efd21e866b9cb3466cb65b963933f5e98016d1",
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-14B",
        default_lr=1e-5,
        eval_batch_size=32,
        gradient_checkpointing=True,
        model_parallel=True,
        # note: you will probably not be able to run this bf16 support and without many gpus
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
            "revision": "8be2854218fea9054331e217fd26a06f3fd02004",
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-72B",
        default_lr=1e-5,
        eval_batch_size=1,
        gradient_checkpointing=True,
        model_parallel=True,
        # note: you will probably not be able to run this without bf16 support and many gpus
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
            "revision": "fec78c0e3b3b10dd9f0ce775c34a686a3255a7d1",
        },
        # This model is really big, save space by using adafactor.
        # Note that even then it will take up ~60GB per GPU on an 8-GPU machine.
        default_optimizer="adafactor",
    ),
]


MODELS_DICT: Dict[str, ModelConfig] = {
    model_config.name: model_config for model_config in MODEL_CONFIGS
}


loss_dict = {
    "logconf": logconf_loss_fn(),
    "product": product_loss_fn(),
    "xent": xent_loss(),
    "conf_induc": conf_induc_loss(),
    "conf_induc_filt": conf_induc_filt_loss()
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
    n_valid_docs: int = 500,
    n_test_docs: int = 1000,
    model_size: str = "gpt2-large",
    lr: Optional[float] = None,
    optim: Optional[str] = None,
    epochs: int = 2,
    force_retrain: bool = False,
    seed: int = 0,
    minibatch_size_per_device: Optional[float] = None,
    train_with_dropout: bool = False,
    results_folder: str = "./weak-to-strong/results",
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
    eval_every: int = 60,
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


    # File directory name configuration
    wms_4_file = weak_model_size.replace("/", "") if weak_model_size is not None else weak_model_size
    ms_4_file = model_size.replace("/", "") if model_size is not None else model_size

    result_dir = "./weak-to-strong/results"
    shared_info_file_dir = result_dir + f"/sample_difficulty/seed={seed}/wms:{wms_4_file}_ms:{ms_4_file}"
    if wms_4_file is not None:
        os.makedirs(shared_info_file_dir, exist_ok=True)


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
        "train_mode": "simple"
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

    # Load dataset                                                                                               # 다음과 같이 수정 필요
    dataset = load_dataset(ds_name, seed=seed, split_sizes=dict(train=n_docs, test=n_test_docs))  # train=n_docs+n_valid_docs

    # Split the training dataset in half
    train_dataset, test_ds = dataset["train"], dataset["test"]                 

    if weak_labels_path is None:                                                    
        # split train and validation set
        train_val_ds = train_dataset.train_test_split(test_size=n_valid_docs, seed=seed)  
        train_ds, valid_ds = train_val_ds["train"], train_val_ds["test"]

        split_data = train_ds.train_test_split(test_size=0.5, seed=seed)       
        train1_ds, train2_ds = split_data["train"], split_data["test"]              
        print("len(train1):", len(train1_ds), "len(train2):", len(train2_ds))  
    
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
       

        # validation set
        valid_ds_dir = os.path.join(os.path.dirname(weak_labels_path), "valid_ds")
        valid_ds = load_from_disk(valid_ds_dir)

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
    valid_ds = tokenize_dataset(valid_ds, tokenizer, max_ctx)
    test_ds = tokenize_dataset(test_ds, tokenizer, max_ctx)
    if train2_ds:
        train2_ds = tokenize_dataset(train2_ds, tokenizer, max_ctx)
    train1_ds.save_to_disk(os.path.join(save_path, 'train_ds/')) 
    loss_fn = loss_dict[loss]
    n_docs = len(train1_ds)

    
    # Train and evaluation
    print(f"Training model model, size {model_size}")
    test_results, inference_results, valid_ds = train_and_save_model(
        model_config,
        train1_ds,
        valid_ds,
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

    # Save datasets
    # valid dataset
    if valid_ds is not None:
        valid_ds.save_to_disk(os.path.join(save_path, "valid_ds"))

    # weak labels
    if inference_results is not None:      
        save_path_wl = save_path + "/" + "weak_labels"
        inference_results.save_to_disk(save_path_wl)

    # test results
    if test_results is not None:
        test_results.save_to_disk(save_path)

        acc = np.mean([x["acc"] for x in test_results])
        res_dict = {"accuracy": acc}
        print("accuracy:", acc)

        with open(os.path.join(save_path, f"results_summary.json"), "w") as f:
            json.dump(res_dict, f, indent=2)
    
    with open(os.path.join(save_path, f"config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("Files are saved")

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

    shared_acts_dir = Path(f"./weak-to-strong/activations/{ds_name}/n{n_docs}/seed={seed}")
    
    # Caching activations 
    if weak_labels_path is None:
        cfg = SFTConfig(
            dataset=ds_name,
            model_name = model_size,
            n_train=len(train1_ds),             
            n_val=0,                                  
            n_test=len(test_ds),
            n_predict=0,
            minibatch_size=1,
            batch_size=32,
            results_folder=shared_acts_dir,   # results_folder 수정 필요
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
            evaluation_strategy="steps",
            label_names=["labels"],
            load_best_model_at_end=cfg.load_best_model_at_end,
            logging_steps=25,
            metric_for_best_model=cfg.metric_for_best_model,
            greater_is_better=cfg.greater_is_better,
            per_device_train_batch_size=cfg.minibatch_size,
            per_device_eval_batch_size=cfg.minibatch_size,
            save_strategy="steps",
            save_total_limit=cfg.save_total_limit,
            #tf32=False,  # Use Tensor Cores even for fp32 matmuls
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
        model_cfg, run_name = get_model_and_run_name(cfg.model_name, "weak")
        train_args["run_name"] = run_name
        train_args["output_dir"] = str(shared_root / cfg_name / "output")
        train_args["learning_rate"] = cfg.weak_lr


        act_ds_dict = DatasetDict(
            {
                "probe_train": train1_ds,         # Already tokenized above
                "target": train2_ds,
            }
        )
        
        acts_dir = shared_acts_dir / f"ms:{ms_4_file}"
        print("Start caching activation")
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

        weak_acts_dir = shared_acts_dir / f"ms:{wms_4_file}"
        strong_acts_dir = shared_acts_dir / f"ms:{ms_4_file}"

        x_weak_fit, weak_fit_idx = load_activations(weak_acts_dir / "probe_train.pt")
        x_weak_tgt, weak_tgt_idx = load_activations(weak_acts_dir / "target.pt")
        x_strong_tgt, strong_tgt_idx = load_activations(strong_acts_dir / "target.pt")

        gt_by_idx = {
            int(i): int(l) for i, l in zip(train_dataset["idx"], train_dataset["hard_label"])
        }

        fit_common = np.array(
            sorted(set(weak_fit_idx.tolist()) & set(gt_by_idx)), dtype=np.int64
        )
        
        wf_pos = {int(v): i for i, v in enumerate(weak_fit_idx)}
        X_fit = x_weak_fit[[wf_pos[int(i)] for i in fit_common]]
        y_fit = torch.tensor([gt_by_idx[int(i)] for i in fit_common], device=X_fit.device)

        weak_probe = PROBES[probe_name](probe_cfg)
        weak_probe.fit(X_fit, y_fit)
        print(f"Probe fitted on {len(fit_common)} samples (GT half)")

        labeled_idx = set(int(i) for i in train1_ds["idx"])
        common = np.array(
            sorted(set(weak_tgt_idx.tolist()) & set(strong_tgt_idx.tolist()) & labeled_idx),
            dtype=np.int64,
        )
        print(
            f"weak tgt: {len(weak_tgt_idx)}, strong tgt: {len(strong_tgt_idx)}, "
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

        wt_pos = {int(v): i for i, v in enumerate(weak_tgt_idx)}
        st_pos = {int(v): i for i, v in enumerate(strong_tgt_idx)}
        x_weak = x_weak_tgt[[wt_pos[int(i)] for i in common]]
        x_strong = x_strong_tgt[[st_pos[int(i)] for i in common]]

        p = weak_probe.predict(x_weak).detach().float().cpu().numpy().reshape(-1)

        conf = 2 * np.abs(p - 0.5)
        sorted_conf = np.sort(conf)
        cp = Binseg(model="l2").fit(sorted_conf.reshape(-1, 1)).predict(n_bkps=1)[0]
        conf_thr = sorted_conf[min(cp, len(sorted_conf) - 1)]

        hard_mask = conf <= conf_thr
        rest_mask = ~hard_mask
        if hard_mask.sum() == 0 or rest_mask.sum() == 0:
            raise RuntimeError(f"Degenerate confidence split: hard={hard_mask.sum()}")

        x_strong_np = x_strong.detach().float().cpu().numpy()
        Xh = x_strong_np[hard_mask]
        Xr = x_strong_np[rest_mask]
        Xh_n = Xh / np.linalg.norm(Xh, axis=1, keepdims=True)
        Xr_n = Xr / np.linalg.norm(Xr, axis=1, keepdims=True)
        align = np.abs(Xr_n @ Xh_n.T).max(axis=1)

        sorted_align = np.sort(align)
        cp2 = Binseg(model="l2").fit(sorted_align.reshape(-1, 1)).predict(n_bkps=1)[0]
        align_thr = sorted_align[min(cp2, len(sorted_align) - 1)]

        hard_idx = common[hard_mask]
        rest_idx = common[rest_mask]
        overlap_idx = rest_idx[align >= align_thr]
        easy_idx = rest_idx[align < align_thr]

        print(f"easy={len(easy_idx)}, overlap={len(overlap_idx)}, hard={len(hard_idx)}")

        sample_diff_dict = {
            "idx": np.concatenate([easy_idx, overlap_idx, hard_idx]).tolist(),
            "difficulty_label": [0] * len(easy_idx)
            + [1] * len(overlap_idx)
            + [2] * len(hard_idx),
        }

        # Save the classification results
        print("========== Save the classification results ==========\n")
        pd.DataFrame(sample_diff_dict).to_csv(
            os.path.join(shared_info_file_dir, "sample_info_label.csv"), index=False
        )

        # Save the classified dataset
        diff_ds_prnt_dir = result_dir + f"/diff_ds/seed={seed}"
        idx_to_pos = {int(v): i for i, v in enumerate(train1_ds["idx"])}

        for diff, idx_set in dict(
            easy=easy_idx, overlap=overlap_idx, hard=hard_idx
        ).items():
            missing = [int(i) for i in idx_set if int(i) not in idx_to_pos]
            assert not missing, f"{len(missing)} idx missing from train1_ds: {missing[:5]}"

            positions = [idx_to_pos[int(i)] for i in idx_set]
            ds = train1_ds.select(positions)
            ds = ds.add_column("difficulty", [1 if diff == "hard" else 0] * len(ds))

            diff_ds_dir = diff_ds_prnt_dir + f"/wms:{wms_4_file}_ms:{ms_4_file}/{diff}_ds"
            ds.save_to_disk(diff_ds_dir)


if __name__ == "__main__":
    fire.Fire(main)
