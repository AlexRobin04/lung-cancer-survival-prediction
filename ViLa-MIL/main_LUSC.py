from __future__ import print_function

import argparse
import logging
import os
from datetime import datetime

# 抑制 PyTorch NNPACK CPU 警告刷屏（不影响训练，仅过滤 WARNING 日志）
os.environ["TORCH_LOGS"] = "+ERROR"

import numpy as np
import pandas as pd
import torch

from datasets.dataset_generic import Generic_MIL_Dataset
from utils.core_utils import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed=7):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description="Configurations for MIL / patch-feature training")
parser.add_argument("--cancer", type=str, default="LUSC", help="the type of cancer")
parser.add_argument("--data_root_dir", type=str, default=".", help="data directory")
parser.add_argument("--data_folder_s", type=str, default="features/20", help="dir under data directory")
parser.add_argument("--data_folder_l", type=str, default="features/10", help="dir under data directory")
parser.add_argument("--max_epochs", type=int, default=100, help="maximum number of epochs")
parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
parser.add_argument("--label_frac", type=float, default=1.0, help="fraction of training labels")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--k", type=int, default=4, help="number of folds")
parser.add_argument("--k_start", type=int, default=-1, help="start fold")
parser.add_argument("--k_end", type=int, default=-1, help="end fold")
parser.add_argument("--results_dir", default="result", help="results directory")
parser.add_argument("--split_dir", type=str, default="splits/TCGA_LUSC")
parser.add_argument("--log_data", action="store_true", default=False, help="log data using tensorboard")
parser.add_argument("--testing", action="store_true", default=False, help="debugging tool")
parser.add_argument("--hard_or_soft", default=False, help="False_hard; True_soft")
parser.add_argument("--early_stopping", action="store_true", default=False, help="enable early stopping")
parser.add_argument("--opt", type=str, choices=["adam", "sgd"], default="adam")
parser.add_argument("--drop_out", action="store_true", default=True, help="enable dropout")
parser.add_argument(
    "--model_type",
    type=str,
    choices=["ViLa_MIL", "TransMIL", "AMIL", "WiKG", "RRTMIL", "PatchGCN", "surformer", "DSMIL", "S4MIL", "EnsembleFeature"],
    default="RRTMIL",
    help="type of model",
)
parser.add_argument("--mode", type=str, choices=["transformer"], default="transformer")
parser.add_argument("--exp_code", type=str, help="experiment code for saving results")
parser.add_argument("--weighted_sample", action="store_true", default=False, help="enable weighted sampling")
parser.add_argument("--reg", type=float, default=1e-5, help="weight decay")
parser.add_argument("--bag_loss", type=str, choices=["svm", "ce", "focal"], default="ce")
parser.add_argument("--task", default="task_tcga_lusc_subrisk", type=str)
parser.add_argument("--text_prompt", type=str, default=None)
parser.add_argument("--text_prompt_path", type=str, default="text_prompt/TCGA_lusc_two_scale_text_prompt.csv")
parser.add_argument("--prototype_number", type=int, default=16)
parser.add_argument(
    "--conch_checkpoint_path",
    type=str,
    default="",
    help="path to conch.pth; empty means use CONCH_CKPT_PATH or ckpt/conch.pth",
)
parser.add_argument("--freeze_base", action="store_true", default=True, help="freeze 5 baselines when training EnsembleFeature")
parser.add_argument(
    "--finetune_ensemble",
    action="store_true",
    default=False,
    help="EnsembleFeature: unfreeze the 5 baseline MIL modules for end-to-end fine-tuning",
)
parser.add_argument(
    "--ensemble_ckpt_dir",
    type=str,
    default=None,
    help="optional: directory with 5 baseline .pt files (filename contains RRTMIL/AMIL/...); if unset, auto-resolve from uploaded_features/best_models.json + tasks.json",
)
parser.add_argument(
    "--ensemble_best_models_json",
    type=str,
    default=None,
    help="optional: override path to best_models.json for EnsembleFeature auto ckpt",
)
parser.add_argument(
    "--ensemble_tasks_json",
    type=str,
    default=None,
    help="optional: override path to tasks.json for EnsembleFeature auto ckpt",
)
parser.add_argument(
    "--ensemble_disable_auto_ckpt",
    action="store_true",
    default=False,
    help="EnsembleFeature: do not auto-resolve baseline checkpoints when ensemble_ckpt_dir is unset",
)

args = parser.parse_args()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if args.text_prompt_path and (not os.path.isabs(args.text_prompt_path)):
    args.text_prompt_path = os.path.join(BASE_DIR, args.text_prompt_path)
args.text_prompt = np.array(pd.read_csv(args.text_prompt_path, header=None)).squeeze()

seed_torch(args.seed)

now = datetime.now()
log_dir = os.path.join(args.results_dir, "logging", "20X", args.cancer, args.model_type)
os.makedirs(log_dir, exist_ok=True)
label = "cancer_={}  lr_={}   max-epoch_={}  drop-out={} hard_or_soft={} in {}-{}-{} {}:{}".format(
    args.cancer, args.lr, args.max_epochs, args.drop_out, args.hard_or_soft, now.year, now.month, now.day, now.hour, now.minute
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(os.path.join(log_dir, label + ".log"), encoding="utf-8")
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

settings = {
    "num_splits": args.k,
    "k_start": args.k_start,
    "k_end": args.k_end,
    "task": args.task,
    "max_epochs": args.max_epochs,
    "lr": args.lr,
    "experiment": args.exp_code,
    "label_frac": args.label_frac,
    "seed": args.seed,
    "model_type": args.model_type,
    "mode": args.mode,
    "use_drop_out": args.drop_out,
    "weighted_sample": args.weighted_sample,
    "opt": args.opt,
}

logging.info("\nLoad Dataset")

if args.task == "task_tcga_lusc_subrisk":
    args.n_classes = 4
    dataset = Generic_MIL_Dataset(
        csv_path="datasets_csv/LUSC_new.csv",
        mode=args.mode,
        data_dir_s=os.path.join(args.data_root_dir, args.data_folder_s),
        data_dir_l=os.path.join(args.data_root_dir, args.data_folder_l),
        shuffle=False,
        print_info=False,
        label_dict={"low": 0, "Moderate": 1, "Elevated": 2, "high": 3},
        patient_strat=False,
        ignore=[],
    )
else:
    raise NotImplementedError

if args.split_dir is None:
    args.split_dir = os.path.join("splits", args.task + "_{}".format(int(args.label_frac * 100)))
elif not os.path.isabs(args.split_dir):
    args.split_dir = os.path.join(BASE_DIR, args.split_dir)

logging.info("split_dir: {}".format(args.split_dir))
assert os.path.isdir(args.split_dir)

settings.update({"split_dir": args.split_dir})

logging.info("################# Settings ###################")
for key, val in settings.items():
    logging.info("{}:  {}".format(key, val))


def main(args_):
    start = 0 if args_.k_start == -1 else args_.k_start
    end = args_.k if args_.k_end == -1 else args_.k_end

    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args_.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(
            from_id=False, csv_path="{}/splits_{}.csv".format(args_.split_dir, i)
        )
        datasets = (train_dataset, val_dataset, test_dataset)
        train(datasets, i, args_)


if __name__ == "__main__":
    main(args)

