import gc
import os
import json
import torch
from typing import Dict

import numpy as np
from ruptures import Binseg
from pathlib import Path
from datasets import DatasetDict, load_from_disk
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import load_sharded_checkpoint

from weak_to_strong.datasets import load_dataset, tokenize_dataset
from weak_to_strong.common import get_tokenizer
from weak_to_strong.train import ModelConfig
from weak_to_strong.model import TransformerWithHead

from datacentric.sft import load_activations
from datacentric.sft_utils import (
    clear_mem,
    gather_hiddens,
)


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
    )
]


MODELS_DICT: Dict[str, ModelConfig] = {
    model_config.name: model_config for model_config in MODEL_CONFIGS
}


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


def load_ckpt(model, ckpt_dir):
    bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    if not os.path.exists(bin_path):
        load_sharded_checkpoint(model, ckpt_dir)
    else:
        sd = torch.load(bin_path, map_location="cpu")
        sd = {k.replace("transformer.module", "transformer"): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)


def build_model(base_name, num_labels, linear_probe, ckpt_dir=None):
    cfg = MODELS_DICT[base_name]
    if cfg.model_parallel:
        model = TransformerWithHead.from_pretrained(
            base_name, 
            num_labels=num_labels,
            device_map="auto",
            linear_probe=linear_probe,
            **(cfg.custom_kwargs or {})
        )
    else:
        model = TransformerWithHead.from_pretrained(
            base_name, 
            num_labels=num_labels,
            linear_probe=linear_probe,
            **(cfg.custom_kwargs or {})
        )

    if ckpt_dir is not None:                  
        load_ckpt(model, ckpt_dir)
        assert model.score.weight.abs().sum() > 0, f"score 미복원: {ckpt_dir}"

    if cfg.model_parallel:
        model.eval()                           # 이미 분산됨, to("cuda") 하지 않음
    else:
        model = model.to("cuda").eval()

    if torch.cuda.device_count() > 1 and not cfg.model_parallel:
        model = torch.nn.DataParallel(model, output_device=0)
    return model


def main(weak_model_size, model_size, strong_loss, seed):
    # hyper-parameters
    ds_name = "cosmos_qa"
    #model_size = "gpt2-xl" 
    #weak_model_size = "gpt2-xl" # weak model size가 gt model 때는 None이고 w2s model 때는 존재하니 if문으로 해결
    loss = "xent"
    #seed = 35
    n_docs = 20000
    n_test_docs = 1000
    max_ctx = 1024
    eval_batch_size = None
    batch_size = 32
    epochs = 2
    minibatch_size_per_device = 1
    train_with_dropout = False
    linear_probe = False
    lr_schedule = "cosine_anneal"
    eval_every = 60

    # num_labels = 2
    # strong_loss = "conf_induc_filt"

    # model config
    weak_model_config = MODELS_DICT[weak_model_size]
    model_config = MODELS_DICT[model_size]
    weak_lr = weak_model_config.default_lr
    lr = model_config.default_lr

    optim = "adam"
    eval_batch_size = weak_model_config.eval_batch_size
    custom_kwargs = weak_model_config.custom_kwargs or {}
    results_folder = "./weak-to-strong/results/train_results"
    sweep_subfolder = f"cosq_large_xent_{seed}/default"

    # config
    config = {
        "batch_size": batch_size,
        "max_ctx": max_ctx,
        "ds_name": ds_name,
        "loss": loss,
        "n_docs": n_docs,
        "n_test_docs": n_test_docs,
        "model_size": weak_model_size,
        "lr": weak_lr,
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

    weak_model_folder_name = get_config_foldername(config)
    weak_model_folder_dir = os.path.join(results_folder, sweep_subfolder, weak_model_folder_name)
    print("ckpt:", weak_model_folder_dir)                                    
    assert os.path.isdir(weak_model_folder_dir), f"폴더 없음: {weak_model_folder_dir}"  

    # load test dataset
    dataset = load_dataset(ds_name, seed=seed, split_sizes=dict(test=n_test_docs))
    test_ds_raw = dataset["test"]

    wk_test_ds_result_dir = os.path.join(weak_model_folder_dir, "test_res")
    wk_test_ds_result = load_from_disk(wk_test_ds_result_dir)



    #######################################
    ###    difficulty classification    ###
    #######################################

    def save_activations(
            shared_acts_dir, 
            ds_dict, 
            model_size, 
            model
        ):
        clear_mem()

        for name, ds in ds_dict.items():
            assert "idx" in ds.column_names, f"{name} split has no 'idx' column"
            acts, idxs = gather_hiddens(model, ds)
            torch.save({"acts": acts.cpu(), "idx": idxs.cpu(), "model": model_size},
                    shared_acts_dir / f"{name}.pt")
            print(f"Saved {len(idxs)} activations for {name}")
    
        torch.cuda.empty_cache()
        gc.collect()


    def group_stats(idx_set, ds):
        pos = {int(v): i for i, v in enumerate(ds["idx"])}

        missing = [int(i) for i in idx_set if int(i) not in pos]
        if missing:
            print(f"[warn] {len(missing)} idx not in ds — dropped")

        sel = [pos[int(i)] for i in idx_set if int(i) in pos]
        accs = np.asarray(ds.select(sel)["acc"], dtype=np.float64)
        return int(accs.sum()), float(accs.mean())

    def sample_difficulty_classification(act_dir, ds):
        # load
        x_strong_w2s, strong_w2s_idx = load_activations(act_dir)

        # align
        idx_pos = {int(v): i for i, v in enumerate(ds["idx"])}   
        target_idx = set(idx_pos)  
        common = np.array(
            sorted(set(strong_w2s_idx.tolist()) & target_idx),  
            dtype=np.int64,
        )
        print(f"strong tgt: {len(strong_w2s_idx)}, ds: {len(target_idx)}, common: {len(common)}")

        if len(common) < len(target_idx):
            print(f"[warn] {len(target_idx) - len(common)} idx missing — stale cache?")
        if len(common) < 10:
            raise RuntimeError("Too few overlapping idx. Regenerate activations?")

        st_pos = {int(v): i for i, v in enumerate(strong_w2s_idx)}
        x_strong = x_strong_w2s[[st_pos[int(i)] for i in common]] 

        # easy, ovlp / hard classification
        p_all = ds.with_format("numpy")["soft_label"]
        print("soft_label:", p_all.dtype, p_all.shape)   
        p = p_all[[idx_pos[int(i)] for i in common]][:, 1]       # p_all shape이 (N, 2)가 아닐 경우 에러 발생
        assert len(p) == len(x_strong) == len(common)   

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
        assert len(align) == len(rest_idx), "len(align) != len(rest_idx)"
        overlap_idx = rest_idx[align >= align_thr]
        easy_idx = rest_idx[align < align_thr]

        return easy_idx, overlap_idx, hard_idx


    # configuration
    wms_4_file = weak_model_size.replace("/", "")
    ms_4_file  = model_size.replace("/", "")

    shared_acts_dir = Path(
        f"./weak-to-strong/activations/{ds_name}/cls/seed={seed}"
        f"/pt_model/ms:{ms_4_file}"
    )
    shared_acts_dir.mkdir(parents=True, exist_ok=True)   

    # save activations
    strong_acts_path = shared_acts_dir / "target.pt"
    if strong_acts_path.exists():
        print(f"[strong] cached: {strong_acts_path}")
    else:
        tok = get_tokenizer(model_size)
        ds_dict = DatasetDict({"target": tokenize_dataset(test_ds_raw, tok, max_ctx)})

        m = AutoModelForCausalLM.from_pretrained(
            model_size, **(MODELS_DICT[model_size].custom_kwargs or {})
        ).to("cuda").eval()
        save_activations(shared_acts_dir, ds_dict, model_size, m)

        del m; torch.cuda.empty_cache(); gc.collect()


    easy_idx, overlap_idx, hard_idx = sample_difficulty_classification(strong_acts_path, wk_test_ds_result)

    # length of datasets
    num_easy = len(easy_idx)
    num_ovlp = len(overlap_idx)
    num_hard = len(hard_idx)
    num_total = num_easy + num_ovlp + num_hard

    print(f"easy={num_easy}, overlap={num_ovlp}, hard={num_hard}")

    # calculate statistics
    corr_easy, acc_easy = group_stats(easy_idx, wk_test_ds_result)   # corr는 weak teacher model이 맞춘 수
    corr_ovlp, acc_ovlp = group_stats(overlap_idx, wk_test_ds_result)
    corr_hard, acc_hard = group_stats(hard_idx, wk_test_ds_result)
    corr_total = corr_easy + corr_ovlp + corr_hard
    acc_total = corr_total / num_total

    # save classification results
    wk_tchr_diff_info = {
        "num_easy": num_easy,
        "num_overlap": num_ovlp,
        "num_hard": num_hard,
        "%_easy": num_easy / num_total,
        "%_overlap": num_ovlp / num_total,
        "%_hard": num_hard / num_total,
        "num_total": num_total,
        "corr_easy": corr_easy,
        "corr_ovlp": corr_ovlp,
        "corr_hard": corr_hard,
        "acc_easy": acc_easy,
        "acc_overlap": acc_ovlp,
        "acc_hard": acc_hard,
        "acc_total": acc_total,
    }

    test_ds_diff_dir = Path(
        f"./weak-to-strong/results/sample_difficulty/seed={seed}/test_ds"
        f"/wms:{wms_4_file}_ms:{ms_4_file}/loss={strong_loss}")
    test_ds_diff_dir.mkdir(parents=True, exist_ok=True)

    with open(test_ds_diff_dir / "weak_teacher_diff_info.json", "w") as f:
        json.dump(wk_tchr_diff_info, f, indent=2)


    # strong student test dataset result

    # strong config
    strong_config = {
        "batch_size": batch_size,
        "max_ctx": max_ctx,
        "ds_name": ds_name,
        "loss": strong_loss,
        "n_docs": n_docs,
        "n_test_docs": n_test_docs,
        "model_size": model_size,
        "weak_model_size": weak_model_size,
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
        "train_mode": "simple"  #diff
        # "sweep_subfolder": sweep_subfolder,
    }
    strong_model_folder = get_config_foldername(strong_config)

    # load result
    train_result_dir = Path(f"./weak-to-strong/results/train_results/")
    subfolder_dir = train_result_dir / f"cosq_large_xent_{seed}/default" / strong_model_folder 
    st_stdt_test_ds_dir = subfolder_dir / "test_res" if seed in [20, 30, 35] else subfolder_dir
    # seed 15 : No test_res
    # seed 20 : Yes test_res
    # seed 30 : Yes test_res
    # seed 35 : Yes test_res

    st_test_ds_result = load_from_disk(st_stdt_test_ds_dir)
    st_acts_path = Path(
        f"./weak-to-strong/activations/{ds_name}/cls/seed={seed}/loss={strong_loss}"
        f"/wms:{wms_4_file}_ms:{ms_4_file}/st_target.pt"
    )

    # calculation
    st_easy_idx, st_overlap_idx, st_hard_idx = sample_difficulty_classification(
        strong_acts_path,
        st_test_ds_result
    )

    # length of datasets
    st_num_easy = len(st_easy_idx)
    st_num_ovlp = len(st_overlap_idx)
    st_num_hard = len(st_hard_idx)
    st_num_total = st_num_easy + st_num_ovlp + st_num_hard

    print(f"easy={st_num_easy}, overlap={st_num_ovlp}, hard={st_num_hard}")

    # calculate statistics
    corr_easy, acc_easy = group_stats(st_easy_idx, st_test_ds_result)   # corr는 weak teacher model이 맞춘 수
    corr_ovlp, acc_ovlp = group_stats(st_overlap_idx, st_test_ds_result)
    corr_hard, acc_hard = group_stats(st_hard_idx, st_test_ds_result)
    corr_total = corr_easy + corr_ovlp + corr_hard
    acc_total = corr_total / st_num_total

    st_stdt_diff_info = {
        "num_easy": st_num_easy,
        "num_overlap": st_num_ovlp,
        "num_hard": st_num_hard,
        "%_easy": st_num_easy / st_num_total,
        "%_overlap": st_num_ovlp / st_num_total,
        "%_hard": st_num_hard / st_num_total,
        "num_total": st_num_total,
        "num_corr_easy": corr_easy,
        "num_corr_ovlp": corr_ovlp,
        "num_corr_hard": corr_hard,
        "acc_easy": acc_easy,
        "acc_overlap": acc_ovlp,
        "acc_hard": acc_hard,
        "acc_total": acc_total,
        "loss": strong_loss,
    }

    with open(test_ds_diff_dir / "strong_student_result_info.json", "w") as f:
        json.dump(st_stdt_diff_info, f, indent=2)



    # classify sample difficulty on weak teacher perspective
    # A result of a strong student model on test set 
    # The numbers of samples for each sample difficulty are presented in the weak model prespective
    # The #correct samples and accuracy are presented in the storng student model.

    corr_easy, acc_easy = group_stats(easy_idx, st_test_ds_result)  
    corr_ovlp, acc_ovlp = group_stats(overlap_idx, st_test_ds_result)
    corr_hard, acc_hard = group_stats(hard_idx, st_test_ds_result)
    corr_total = corr_easy + corr_ovlp + corr_hard
    acc_total = corr_total / num_total

    st_stdt_diff_info_diff_wk = {
        "num_easy": num_easy,                  # weak model perspective
        "num_overlap": num_ovlp,
        "num_hard": num_hard,
        "%_easy": num_easy / num_total,
        "%_overlap": num_ovlp / num_total,
        "%_hard": num_hard / num_total,
        "num_total": num_total,
        "num_corr_easy": corr_easy,            # strong model perspective
        "num_corr_ovlp": corr_ovlp,
        "num_corr_hard": corr_hard,
        "acc_easy": acc_easy,
        "acc_overlap": acc_ovlp,
        "acc_hard": acc_hard,
        "acc_total": acc_total,
        "loss": strong_loss,
    }

    with open(test_ds_diff_dir / "strong_student_result_in_wk_perspec.json", "w") as f:
        json.dump(st_stdt_diff_info_diff_wk, f, indent=2)


    # strong ceiling model perspective cls
    # strong student model result

    ceil_config = {
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
    ceil_model_folder_name = get_config_foldername(ceil_config)
    ceil_test_ds_result_dir = os.path.join(results_folder, sweep_subfolder, ceil_model_folder_name, "test_res")

    ceil_test_ds_result = load_from_disk(ceil_test_ds_result_dir)
    ceil_acts_path = strong_acts_path
    ceil_easy_idx, ceil_overlap_idx, ceil_hard_idx = sample_difficulty_classification(
        ceil_acts_path,
        ceil_test_ds_result
    )

    # length of datasets
    ceil_num_easy = len(ceil_easy_idx)
    ceil_num_ovlp = len(ceil_overlap_idx)
    ceil_num_hard = len(ceil_hard_idx)
    ceil_num_total = ceil_num_easy + ceil_num_ovlp + ceil_num_hard

    print(f"easy={ceil_num_easy}, overlap={ceil_num_ovlp}, hard={ceil_num_hard}")

    corr_easy, acc_easy = group_stats(ceil_easy_idx, st_test_ds_result)  
    corr_ovlp, acc_ovlp = group_stats(ceil_overlap_idx, st_test_ds_result)
    corr_hard, acc_hard = group_stats(ceil_hard_idx, st_test_ds_result)
    corr_total = corr_easy + corr_ovlp + corr_hard
    acc_total = corr_total / ceil_num_total

    ceil_stdt_diff_info_diff_ceil = {
        "num_easy": ceil_num_easy,                  
        "num_overlap": ceil_num_ovlp,
        "num_hard": ceil_num_hard,
        "%_easy": ceil_num_easy / ceil_num_total,
        "%_overlap": ceil_num_ovlp / ceil_num_total,
        "%_hard": ceil_num_hard / ceil_num_total,
        "num_total": ceil_num_total,
        "num_corr_easy": corr_easy,           
        "num_corr_ovlp": corr_ovlp,
        "num_corr_hard": corr_hard,
        "acc_easy": acc_easy,
        "acc_overlap": acc_ovlp,
        "acc_hard": acc_hard,
        "acc_total": acc_total,
        "loss": strong_loss,
    }

    with open(test_ds_diff_dir / "strong_student_result_in_ceil_perspec.json", "w") as f:
        json.dump(ceil_stdt_diff_info_diff_ceil, f, indent=2)

    # 단일 시드 내에서
    # weak teacher model 관점에서 test set의 difficulty distribution을 확인할 수 있음
    # json 파일 안의 acc 정보도 weak model 기준

    # strong student model 관점에서의 difficulty distribution of a test set도 궁금함

    return


if __name__ == "__main__":
    seed = 20
    models = ["gpt2-large", "gpt2-xl", "Qwen/Qwen-1_8B"]
    losses = ["xent", "conf_induc_filt", "conf_induc"]

    for n_ls in range(len(losses)):
        for i_wk in range(len(models)):
            for j_st in range(i_wk, len(models)):
                main(models[i_wk], models[j_st], losses[n_ls], seed)

