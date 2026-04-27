from __future__ import print_function

import argparse
import logging
import os
import warnings
from datetime import datetime

# TORCH_LOGS 只能是 PyTorch 文档中的模块/artifact 名列表；非法值（如 +ERROR）会在 import torch 时抛 ValueError。
os.environ.pop("TORCH_LOGS", None)

# NNPACK 等 CPU 后端提示：不影响正确性，仅减少 stderr/训练日志刷屏（须在 import torch 之前注册）。
_nnpack_msg = r"(?is).*(?:\bnnpack\b|could not initialize nnpack|nnpack is not supported|compiled without nnpack).*"
for _cat in (UserWarning, RuntimeWarning):
    warnings.filterwarnings("ignore", message=_nnpack_msg, category=_cat)

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
    choices=[
        "ViLa_MIL",
        "TransMIL",
        "AMIL",
        "WiKG",
        "RRTMIL",
        "PatchGCN",
        "surformer",
        "DSMIL",
        "S4MIL",
        "EnsembleDecision",
    ],
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
    help="optional: override path to best_models.json for EnsembleDecision 五路基线自动解析",
)
parser.add_argument(
    "--ensemble_tasks_json",
    type=str,
    default=None,
    help="optional: override path to tasks.json for EnsembleDecision 五路基线自动解析",
)
parser.add_argument(
    "--ensemble_disable_auto_ckpt",
    action="store_true",
    default=False,
    help="EnsembleDecision: 未指定 ensemble_ckpt_dir 时不从 best_models/tasks 自动解析五路基线权重",
)
parser.add_argument(
    "--ensemble_exclude",
    type=str,
    default="",
    help="EnsembleDecision: 逗号分隔要排除的基线；不可写满五路",
)
parser.add_argument(
    "--ensemble_branch_prior",
    type=str,
    default="",
    help="EnsembleDecision: 分支先验（如队列 C-index），用于 weighted 固定融合权重",
)
parser.add_argument(
    "--ensemble_branch_prior_scale",
    type=float,
    default=1.25,
    help="EnsembleDecision: 先验 logit 强度系数",
)
parser.add_argument(
    "--ensemble_branch_prior_temperature",
    type=float,
    default=1.0,
    help="EnsembleDecision: 有 branch_prior 时先验 logit 除以该温度",
)
parser.add_argument(
    "--decision_fusion",
    type=str,
    choices=["avg_prob"],
    default="avg_prob",
    help=(
        "EnsembleDecision: avg_prob=各基线输出概率逐类平均（简单可解释，默认推荐）。"
    ),
)
parser.add_argument(
    "--decision_branch_weights",
    type=str,
    default="",
    help=(
        "保留参数（兼容旧任务），当前 avg_prob 融合不使用该字段。"
    ),
)
parser.add_argument(
    "--ensemble_auto_tune_weights",
    type=lambda x: str(x).strip().lower() in {"1", "true", "yes", "y", "on"},
    default=True,
    help="EnsembleDecision(weighted): 是否在验证集上自动搜索分支权重（优先 C-index）",
)
parser.add_argument(
    "--ensemble_weight_search_trials",
    type=int,
    default=256,
    help="EnsembleDecision 自动调权随机采样次数（Dirichlet 试探）",
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
if args.model_type == "EnsembleDecision":
    settings["decision_fusion"] = args.decision_fusion
    if getattr(args, "decision_branch_weights", "").strip():
        settings["decision_branch_weights"] = args.decision_branch_weights.strip()
    settings["ensemble_auto_tune_weights"] = bool(args.ensemble_auto_tune_weights)
    settings["ensemble_weight_search_trials"] = int(args.ensemble_weight_search_trials)

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

