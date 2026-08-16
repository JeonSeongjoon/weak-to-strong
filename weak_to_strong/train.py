import itertools
import os
import pickle
import time
import json
from dataclasses import dataclass
from typing import Callable, Optional

import datasets
import numpy as np
import pandas as pd
import torch
import torch_optimizer as toptim
from transformers.modeling_utils import load_sharded_checkpoint

import weak_to_strong.logger as logger
from weak_to_strong.common import clear_mem
from weak_to_strong.eval import eval_model_acc
from weak_to_strong.loss import xent_loss
from weak_to_strong.model import TransformerWithHead


@dataclass
class ModelConfig:
    name: str
    default_lr: float
    eval_batch_size: int
    custom_kwargs: Optional[dict] = None
    gradient_checkpointing: bool = False
    model_parallel: bool = False
    default_optimizer: str = "adam"


def train_model(
    model: torch.nn.Module,
    ds: datasets.Dataset,
    valid_ds: datasets.Dataset,
    batch_size: int,
    lr: float = 1e-5,
    loss_fn: Callable = None,
    log_every: int = 10,
    eval_every: int = 100,
    eval_batch_size: int = 256,
    minibatch_size: int = 8,
    eval_ds: Optional[datasets.Dataset] = None,
    gradient_checkpointing: bool = False,
    train_with_dropout: bool = False,
    epochs: int = 1,
    lr_schedule: str = "cosine_anneal",
    optimizer_name: str = "adam"
):
    print("LR", lr, "batch_size", batch_size, "minibatch_size", minibatch_size)
    assert batch_size % minibatch_size == 0, "batch size must be divisible by minibatch size"
    # we purposefully turn off dropout, for determinism
    # this seems to help for 1 epoch finetuning anyways
    if train_with_dropout:
        model.train()
    else:
        model.eval()
        
    if gradient_checkpointing:
        (
            model if hasattr(model, "gradient_checkpointing_enable") else model.module
        ).gradient_checkpointing_enable()

    nsteps = len(ds) * epochs // batch_size

    def lr_schedule_fn(step):
        if lr_schedule == "constant":
            return 1
        else:
            assert False, f"invalid lr schedule, {lr_schedule}, must be constant or cosine_anneal"

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "adafactor":
        optimizer = toptim.Adafactor(model.parameters(), lr=lr)
    else:
        assert False, f"invalid optimizer {optimizer_name}, must be adam or adafactor"

    if lr_schedule == "cosine_anneal":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, nsteps)
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule_fn)
                       
    step = 0
    saving_interval = 300
    stop_steps = 490               # 490
    # stop training when "step" becomes a certain number.
    final_eval_results = None
    best_loss = 100

    it = itertools.chain.from_iterable(itertools.repeat(ds, epochs))
    losses = []

    accuracies = []
    thresholds = {}                             
    sample_info = {}     
    is_conf_induc_anc = loss_fn.name.startswith("conf_induc_anc")                  
    is_conf_induc = loss_fn.name.startswith("conf_induc") and not is_conf_induc_anc
    
    

    # If the model is wrapped by DataParallel, it doesn't have a device. In this case,
    # we use GPU 0 as the output device. This sadly means that this device will store
    # a bit more data than other ones, but hopefully should not be too big of a deal.
    io_device = model.device if hasattr(model, "device") else 0

    while step < nsteps and step <= stop_steps:
        loss_tot = 0

        if eval_every and (step + 1) % eval_every == 0:
            eval_results, gold_loss, weak_loss = eval_model_acc(model, valid_ds, eval_batch_size)  
            if gradient_checkpointing:
                (
                    model if hasattr(model, "gradient_checkpointing_enable") else model.module
                ).gradient_checkpointing_enable()
            if train_with_dropout:
                model.train()
            gold_acc = np.mean([r["gt_acc"] for r in eval_results])
            weak_acc = np.mean([r["acc"] for r in eval_results])

            # In gt model training, valid_loss_gd, valid_loss will be same. gt model has no weak label!
            logger.logkvs(
                {
                    "valid_accuracy_gd": gold_acc,
                    "valid_loss_gd": gold_loss,
                    "valid_accuracy_wk": weak_acc,
                    "valid_loss_wk": weak_loss,
                }
            )

            # nsteps should be bigger than saving_interval. If not, model would not save the final_eval_results
            if (step > saving_interval) and gold_loss < best_loss:
                print("Evaluation : the best valid loss model")
                best_loss = gold_loss
                final_eval_results, _, _ = eval_model_acc(model, eval_ds, eval_batch_size)
                logger.logkv("eval_accuracy", np.mean([r["acc"] for r in final_eval_results]))

        all_logits = []
        all_labels = []
        all_idxs = []
        all_diffs = []

        for i in range(batch_size // minibatch_size):    
            try:
                mbatch = [next(it) for _ in range(minibatch_size)] 
            except StopIteration:
                break

            input_ids = (
                torch.nn.utils.rnn.pad_sequence([torch.tensor(ex["input_ids"]) for ex in mbatch])
                .transpose(
                    0,
                    1,
                )
                .to(io_device)
            )
            labels = torch.tensor([ex["soft_label"] for ex in mbatch]).to(io_device)   # minibatch label set (soft label)
            logits = model(input_ids)

            all_logits.extend(logits.to(io_device))
            all_labels.extend(labels)
            all_idxs.extend([ex["idx"] for ex in mbatch])
            if is_conf_induc_anc: 
                all_diffs.extend([ex["difficulty"] for ex in mbatch])

        all_logits = torch.stack(all_logits)
        all_labels = torch.stack(all_labels)
        all_diffs = torch.tensor(all_diffs, dtype=torch.float32, device=io_device) if is_conf_induc_anc else None

        loss = loss_fn(all_logits, all_labels, step_frac=step/nsteps, diff=all_diffs)
        loss_tot += loss.item()
        loss.backward()
        losses.append(loss_tot)


        if is_conf_induc:
            if not sample_info:
                for key in ["idx", "difficulty", "confidence"]:
                    sample_info[key] = []

            # logging threshold value for each step
            thresholds[f"step{step}"] = loss_fn.threshold          

            # logging sample info for each sample
            sample_es_or_olp = loss_fn.easy 
            sample_conf = loss_fn.conf
            for i in range(len(all_idxs)):
                diff_val = sample_es_or_olp[i]            
                sample_info["idx"].append(all_idxs[i])
                sample_info["difficulty"].append(diff_val.item() if hasattr(diff_val, "item") else diff_val)
                sample_info["confidence"].append(sample_conf[i].item())


        accuracies.append(
            torch.mean(
                (torch.argmax(all_logits, dim=1) == torch.argmax(all_labels, dim=1)).to(
                    torch.float32
                )
            ).item()
        )
        logger.logkvs(
            {
                "step": step,
                "progress": step / nsteps,
                "loss": loss_tot,
                "train_accuracy": accuracies[-1],
                "lr": lr_scheduler.get_last_lr()[0],
            }
        )
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        if log_every and step % log_every == 0:
            print(
                f"Step: {step}/{nsteps} Recent losses: {np.mean(losses)} {np.mean(accuracies)} {len(losses)}"
            )
            losses = []
            accuracies = []

        step += 1
        logger.dumpkvs()
    

    return final_eval_results, sample_info, thresholds


def train_and_save_model(
    model_config: ModelConfig,
    train_ds: datasets.Dataset,
    valid_ds: datasets.Dataset,
    test_ds: datasets.Dataset,
    inference_ds: Optional[datasets.Dataset] = None,
    *,
    batch_size: int,
    lr: float,
    epochs: int,
    eval_batch_size: Optional[int] = None,
    minibatch_size_per_device: Optional[int] = None,
    save_path: Optional[str] = None,
    loss_fn: Callable = None,
    label: str = "default",
    force_retrain: bool = False,
    train_with_dropout: bool = False,
    linear_probe: bool = False,
    lr_schedule: str = "constant",
    optimizer_name: str = "adam",
    eval_every: Optional[int] = None,
    weak_model_size: Optional[str] = None,
    shared_info_file_dir: str = None
):
    if eval_batch_size is None:
        eval_batch_size = batch_size

    if minibatch_size_per_device is None:
        minibatch_size_per_device = 1

    gradient_checkpointing = model_config.gradient_checkpointing
    custom_kwargs = model_config.custom_kwargs or {}

    def maybe_load_model(model):
        if os.path.exists(os.path.join(save_path, "results.pkl")) and not force_retrain:
            return True
            #print("loading from", save_path)
            #checkpoint_path = os.path.join(save_path, "pytorch_model.bin")
            #index_path = os.path.join(save_path, "pytorch_model.bin.index.json")

            #if not os.path.exists(checkpoint_path):
            #    # Assume this means we have a sharded checkpoint, and load it appropriately
            #    load_sharded_checkpoint(model, save_path)
            #else:
            #    state_dict = torch.load(os.path.join(save_path, "pytorch_model.bin"))
            #    state_dict = {
            #        k.replace("transformer.module", "transformer"): v
            #        for (k, v) in state_dict.items()
            #    }
            #    model.load_state_dict(state_dict, strict=False)
            #return True
        return False

    already_trained = False
    # Load the model
    if model_config.model_parallel:
        assert torch.cuda.device_count() > 1, f"you might want more gpus for {model_config.name}"
        #ngpus = torch.cuda.device_count()
        #max_memory = {i:"20GiB" for i in range(ngpus)}
        model = TransformerWithHead.from_pretrained(
            model_config.name,
            num_labels=2,
            device_map="auto",
            #max_memory=max_memory,
            linear_probe=linear_probe,
            **custom_kwargs,
        )
        already_trained = maybe_load_model(model)
        # slight misnomer, more like minibatch_size_per_dp_replica
        minibatch_size = minibatch_size_per_device
    else:
        model = TransformerWithHead.from_pretrained(
            model_config.name, num_labels=2, linear_probe=linear_probe, **custom_kwargs
        ).to("cuda")
        already_trained = maybe_load_model(model)
        # data parallel:  currently not supported with model parallel

        minibatch_size = min(minibatch_size_per_device * torch.cuda.device_count(), batch_size)

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, output_device=0)
            print(
                "Using",
                torch.cuda.device_count(),
                "GPUs, setting minibatch_size to",
                minibatch_size,
            )
        else:
            minibatch_size = minibatch_size_per_device

    if already_trained:
        test_results = None
        inference_results = None
        valid_ds = None
        # test_results = eval_model_acc(model, test_ds, eval_batch_size)
    else:
        start = time.time()
        test_results, sample_info, thresholds = train_model(
            model,
            train_ds,
            valid_ds,
            batch_size,
            lr=lr,
            epochs=epochs,
            eval_ds=test_ds,
            gradient_checkpointing=gradient_checkpointing,
            loss_fn=loss_fn,
            eval_batch_size=eval_batch_size,
            eval_every=eval_every,
            minibatch_size=minibatch_size,
            train_with_dropout=train_with_dropout,
            lr_schedule=lr_schedule,
            optimizer_name=optimizer_name
        )
        print("Model training took", time.time() - start, "seconds")
        
        if save_path and (weak_model_size is None):  
            # Note: If the model is wrapped by DataParallel, we need to unwrap it before saving
            # Just save the models when they are cases of ground truth training
            #(model if hasattr(model, "save_pretrained") else model.module).save_pretrained(
            #    save_path,
            #    safe_serialization=False
            #)
            print("saved", save_path)

        if (shared_info_file_dir is not None) and (weak_model_size is not None):
            # Save sample info as pandas?
            if sample_info:
                pd.DataFrame(sample_info).to_csv(
                    os.path.join(shared_info_file_dir, 'sample_info.csv'), 
                    index=False
                )
            # Save thresholds dict as json
            if thresholds:
                with open(os.path.join(shared_info_file_dir, "thresholds.json"), "w") as f:
                    json.dump(thresholds, f, indent=2)

        inference_results = None
        if inference_ds:
            inference_results, _, _ = eval_model_acc(model, inference_ds, eval_batch_size)
            logger.logkv("inference_accuracy", np.mean([r["acc"] for r in inference_results]))

    
        if save_path:
            with open(os.path.join(save_path, "results.pkl"), "wb") as f:
                pickle.dump(
                    {
                        "avg_acc_test": float(np.mean([r["acc"] for r in test_results])),
                        "avg_acc_inference": float(
                            np.mean([r["acc"] for r in inference_results] if inference_results else [])
                        ),
                        "test_results": test_results,
                        "inference_results": inference_results if inference_results else [],
                    },
                    f,
                )

        # valid_ds이 이미 weak label화 되었다면, 굳이 다시 X
        if (weak_model_size is None) and (valid_ds is not None):
            valid_ds, _, _ = eval_model_acc(model, valid_ds, eval_batch_size)
        else: 
            valid_ds = None
   
    # try to clean up memory
    clear_mem()
    logger.shutdown()

    return test_results, inference_results, valid_ds
