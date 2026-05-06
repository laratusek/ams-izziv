
import argparse
import json
import torch
import nnunetv2
from typing import Union, Optional
from nnunetv2.run import run_training
from nnunetv2.run.run_finetuning_stunet import load_stunet_pretrained_weights
from unittest.mock import patch

def load_params(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        params = json.load(f)


    assert params.get("device") in ['cpu', 'cuda', 'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {params.get("device")}.'
    if params.get("device") == 'cpu':
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif params.get("device") == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    return {
        "dataset_name_or_id": params["dataset_name_or_id"],
        "configuration": params["configuration"],
        "fold": params["fold"],
        "trainer_class_name": params.get("trainer_class_name", "nnUNetTrainer"),
        "plans_identifier": params.get("plans_identifier", "nnUNetPlans"),
        "pretrained_weights": params.get("pretrained_weights"),
        "num_gpus": params.get("num_gpus", 1),
        "use_compressed_data": params.get("use_compressed_data", False),
        "export_validation_probabilities": params.get(
            "export_validation_probabilities", False
        ),
        "continue_training": params.get("continue_training", False),
        "only_run_validation": params.get("only_run_validation", False),
        "disable_checkpointing": params.get("disable_checkpointing", False),
        "val_with_best": params.get("val_with_best", False),
        "device": device,
    }

def run_training_entry():

    # CLI input
    parser = argparse.ArgumentParser()
    #parser.add_argument('--data_path', type=str, default='', help='') 
    parser.add_argument('-params_path', type=str, default='', help='') #params.json
    #parser.add_argument('--output_path', type=str, default='', help='')

    args = parser.parse_args()
    
    # nastavitve modela
    params = load_params(args.params_path)

    with patch("run_training.load_pretrained_weights", load_stunet_pretrained_weights):
        run_training(**params)
    

if __name__ == '__main__':
    run_training_entry()