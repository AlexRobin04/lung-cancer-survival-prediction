import numpy as np
import torch
from utils.utils import *
import os
from datasets.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.metrics import auc as calc_auc
from utils.loss_utils import FocalLoss


class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        # 支持 tensor / ndarray / python int；统一取标量
        if isinstance(Y_hat, torch.Tensor):
            Y_hat = int(Y_hat.view(-1)[0].item())
        else:
            Y_hat = int(Y_hat)
        if isinstance(Y, torch.Tensor):
            Y = int(Y.view(-1)[0].item())
        else:
            Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        if isinstance(Y_hat, torch.Tensor):
            Y_hat = Y_hat.detach().cpu().view(-1).numpy()
        Y_hat = np.array(Y_hat).astype(int).reshape(-1)
        if isinstance(Y, torch.Tensor):
            Y = Y.detach().cpu().view(-1).numpy()
        Y = np.array(Y).astype(int).reshape(-1)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    elif args.bag_loss == 'focal':
        loss_fn = FocalLoss()
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

    if args.model_type == 'ViLa_MIL':
        import ml_collections
        from models.model_ViLa_MIL import ViLa_MIL_Model
        config = ml_collections.ConfigDict()
        # 尝试从数据样本推断特征维度（不同特征提取器可能是 512/1024 等）
        inferred_input_size = None
        try:
            sample0 = train_split[0]
            x0 = sample0[0] if isinstance(sample0, (tuple, list)) and len(sample0) > 0 else None
            if hasattr(x0, "shape") and len(getattr(x0, "shape")) >= 2:
                inferred_input_size = int(x0.shape[-1])
        except Exception:
            inferred_input_size = None
        config.input_size = inferred_input_size or 512
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        # ViLa_MIL 模型内部会读取该字段；保持与 main_LUSC.py 参数一致
        config.hard_or_soft = bool(args.hard_or_soft)
        model_dict = {'config': config, 'num_classes':args.n_classes}
        model = ViLa_MIL_Model(**model_dict)

    else:
        # Other MIL models (CPU smoke-test friendly: fallback to CE loss if survival labels missing)
        import ml_collections
        config = ml_collections.ConfigDict()
        config.hard_or_soft = bool(getattr(args, "hard_or_soft", False))
        # Infer feature dim from dataset (patch feature width)
        inferred_dim = 512
        try:
            sample0 = train_split[0]
            x0 = sample0[0] if isinstance(sample0, (tuple, list)) and len(sample0) > 0 else None
            if hasattr(x0, "shape") and len(getattr(x0, "shape")) >= 2:
                inferred_dim = int(x0.shape[-1])
        except Exception:
            inferred_dim = 512

        if args.model_type == "RRTMIL":
            from models.RRT import RRTMIL
            model = RRTMIL(config=config, n_classes=args.n_classes, input_dim=inferred_dim)
            model.config = config
        elif args.model_type == "AMIL":
            from models.AMIL import AMIL
            model = AMIL(config=config, n_classes=args.n_classes, input_dim=inferred_dim)
            model.config = config
        elif args.model_type == "WiKG":
            from models.WiKG import WiKG
            model = WiKG(config=config, n_classes=args.n_classes, dim_in=inferred_dim, dim_hidden=512)
            model.config = config
        elif args.model_type == "DSMIL":
            from models.DSMIL import MILNet
            model = MILNet(config=config, in_size=inferred_dim, num_class=args.n_classes, dropout=0.25)
            model.config = config
        elif args.model_type == "S4MIL":
            from models.S4MIL import S4Model
            model = S4Model(
                config=config,
                in_dim=inferred_dim,
                n_classes=args.n_classes,
                dropout=0.1,
                act="relu",
                d_model=512,
                d_state=16,
            )
            model.config = config
        elif args.model_type == "EnsembleFeature":
            from models.EnsembleFeature import EnsembleFeatureMIL
            freeze = bool(getattr(args, "freeze_base", True)) and not bool(getattr(args, "finetune_ensemble", False))
            model = EnsembleFeatureMIL(
                config=config, n_classes=args.n_classes,
                feat_dim=inferred_dim, freeze_base=freeze,
            )
            model.config = config
            # 基线预训练权重：1) 显式目录 ensemble_ckpt_dir；2) 否则从 uploaded_features/best_models.json + tasks.json 按当前折自动解析
            ckpt_dir = getattr(args, "ensemble_ckpt_dir", None)
            ckpt_names = {
                "rrt_ckpt": "RRTMIL",
                "amil_ckpt": "AMIL",
                "wikg_ckpt": "WiKG",
                "dsmil_ckpt": "DSMIL",
                "s4mil_ckpt": "S4MIL",
            }
            ckpt_paths: dict[str, str] = {}
            if ckpt_dir and os.path.isdir(ckpt_dir):
                for key, prefix in ckpt_names.items():
                    for fname in os.listdir(ckpt_dir):
                        if fname.endswith(".pt") and prefix.lower() in fname.lower():
                            ckpt_paths[key] = os.path.join(ckpt_dir, fname)
                            break
                if len(ckpt_paths) == 5:
                    model.load_pretrained(
                        ckpt_paths["rrt_ckpt"],
                        ckpt_paths["amil_ckpt"],
                        ckpt_paths["wikg_ckpt"],
                        ckpt_paths["dsmil_ckpt"],
                        ckpt_paths["s4mil_ckpt"],
                        device=device,
                    )
                    print(f"Loaded pre-trained baselines from {ckpt_dir}")
                else:
                    print(f"Warning: only found {len(ckpt_paths)}/5 baseline checkpoints in {ckpt_dir}")
            elif not getattr(args, "ensemble_disable_auto_ckpt", False):
                from utils.ensemble_ckpt_resolve import auto_resolve_ensemble_pretrained_paths

                vila_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                auto_paths = auto_resolve_ensemble_pretrained_paths(
                    fold_idx=int(cur),
                    cancer=str(getattr(args, "cancer", "LUSC")),
                    mode=str(getattr(args, "mode", "transformer")),
                    vila_root=vila_root,
                    best_models_path=getattr(args, "ensemble_best_models_json", None) or None,
                    tasks_path=getattr(args, "ensemble_tasks_json", None) or None,
                )
                if auto_paths and len(auto_paths) == 5:
                    model.load_pretrained(
                        auto_paths["rrt_ckpt"],
                        auto_paths["amil_ckpt"],
                        auto_paths["wikg_ckpt"],
                        auto_paths["dsmil_ckpt"],
                        auto_paths["s4mil_ckpt"],
                        device=device,
                    )
                    print(
                        f"Loaded pre-trained baselines (auto, fold={cur}) from best_models.json + tasks.json "
                        f"under {os.path.join(vila_root, 'uploaded_features')}"
                    )
                else:
                    print(
                        "Warning: auto baseline checkpoint resolve failed "
                        f"(fold={cur}). Need bestTaskId for RRTMIL/AMIL/WiKG/DSMIL/S4MIL under same cancer/mode "
                        f"and s_{cur}_checkpoint.pt in each task resultsDir. "
                        "Use --ensemble_ckpt_dir or train five baselines via platform first."
                    )
        elif args.model_type == "surformer":
            from models.HVTSurv import HVTSurv
            model = HVTSurv(n_classes=args.n_classes)
            model.config = config
        elif args.model_type in {"TransMIL", "PatchGCN"}:
            # Note: dedicated TransMIL/PatchGCN model files are not present in this codebase.
            # Use MIL baseline as a compatible fallback so API training can run end-to-end.
            if args.n_classes > 2:
                model = MIL_fc_mc(**model_dict)
            else:
                model = MIL_fc(**model_dict)
        else:
            # Fallback: simple MIL baseline
            if args.n_classes > 2:
                model = MIL_fc_mc(**model_dict)
            else:
                model = MIL_fc(**model_dict)


    if hasattr(model, "relocate") and device.type == 'cuda':
        model.relocate()
    else:
        model = model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    # PyTorch 2.0+ 移除了 ReduceLROnPlateau 的 verbose 参数
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=0.1, patience=10, verbose=True
        )
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=0.1, patience=10
        )

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample, mode=args.mode)
    # For per-epoch train metrics (AUC/F1), use a deterministic order loader.
    train_eval_loader = get_split_loader(train_split, training=False, testing=args.testing, weighted=False, mode=args.mode)
    val_loader = get_split_loader(val_split,  testing = args.testing, mode=args.mode)
    test_loader = get_split_loader(test_split, testing = args.testing, mode=args.mode)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=20, stop_epoch=80, verbose=True)
    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        train_loop(args, epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
        stop = validate(cur, epoch, model, val_loader, args.n_classes, 
            early_stopping, writer, loss_fn, args.results_dir)
        if stop: 
            break
        # Compute and print train-set metrics (ROC AUC/F1) for UI summary.
        try:
            _, train_error, train_auc, _, train_f1 = summary(args.mode, model, train_eval_loader, args.n_classes)
            print('Train error: {:.4f}, ROC AUC: {:.4f}, F1: {:.4f}'.format(train_error, train_auc, train_f1))
        except Exception as e:
            # Don't fail training if metrics computation fails; keep a clear log line.
            print(f'Train metrics failed: {e}')

    if args.early_stopping: 
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _, val_f1 = summary(args.mode, model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}, F1: {:.4f}'.format(val_error, val_auc, val_f1))

    results_dict, test_error, test_auc, acc_logger, test_f1 = summary(args.mode, model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}, F1: {:.4f}'.format(test_error, test_auc, test_f1))

    each_class_acc = []
    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        each_class_acc.append(acc)
        acc_str = "N/A" if acc is None else "{:.4f}".format(acc)
        print('class {}: acc {}, correct {}/{}'.format(i, acc_str, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error, each_class_acc, test_f1


def train_loop(args, epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data_s, coord_s, data_l, coords_l, label) in enumerate(loader):
        data_s, coord_s, data_l, coords_l, label = data_s.to(device), coord_s.to(device), data_l.to(device), coords_l.to(device), label.to(device)
        if args.model_type in {"ViLa_MIL", "RRTMIL", "AMIL", "WiKG", "DSMIL", "S4MIL", "surformer", "EnsembleFeature"}:
            logits, Y_prob, loss = model(data_s, coord_s, data_l, coords_l, label)
            Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
            if loss is None:
                loss = loss_fn(logits.view(1, -1), label.view(1))
        else:
            # MIL baselines expect per-instance feature vectors (K, 1024).
            # Our features are two-scale (K, 512) + (K, 512) -> concat to (K, 1024).
            K = min(data_s.size(0), data_l.size(0))
            h = torch.cat([data_s[:K], data_l[:K]], dim=1)
            logits, _, Y_hat, _, _ = model(h)
            loss = loss_fn(logits.view(1, -1), label.view(1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        acc_logger.log(Y_hat, label)
        loss_value = loss.item()
        train_loss += loss_value
        error = calculate_error(Y_hat, label)
        train_error += error

    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        acc_str = "N/A" if acc is None else "{:.4f}".format(acc)
        print('class {}: acc {}, correct {}/{}'.format(i, acc_str, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    all_pred = []
    all_label = []
    with torch.no_grad():
        for batch_idx, (data_s, coord_s, data_l, coords_l, label) in enumerate(loader):
            data_s, coord_s, data_l, coords_l, label = data_s.to(device, non_blocking=True), coord_s.to(device, non_blocking=True), \
                                                                  data_l.to(device, non_blocking=True), coords_l.to(device, non_blocking=True), \
                                                                  label.to(device, non_blocking=True)
            if hasattr(model, 'config') or model.__class__.__name__ in {'ViLa_MIL_Model', 'RRTMIL', 'AMIL', 'WiKG', 'MILNet', 'S4Model', 'HVTSurv'}:
                logits, Y_prob, loss = model(data_s, coord_s, data_l, coords_l, label)
                Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
                if loss is None:
                    loss = loss_fn(logits.view(1, -1), label.view(1))
            else:
                K = min(data_s.size(0), data_l.size(0))
                h = torch.cat([data_s[:K], data_l[:K]], dim=1)
                logits, Y_prob, Y_hat, _, _ = model(h)
                loss = loss_fn(logits.view(1, -1), label.view(1))

            acc_logger.log(Y_hat, label)
            prob[batch_idx] = Y_prob.detach().cpu().view(-1).numpy()
            labels[batch_idx] = label.item()
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            all_pred.append(int(Y_hat.view(-1)[0].item()))
            all_label.append(int(label.view(-1)[0].item()))

    val_error /= len(loader)
    val_loss /= len(loader)
    val_f1 = f1_score(all_label, all_pred, average='macro')

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, f1: {: .4f}'.format(val_loss, val_error, auc, val_f1))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        acc_str = "N/A" if acc is None else "{:.4f}".format(acc)
        print('class {}: acc {}, correct {}/{}'.format(i, acc_str, correct, count))

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_error, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(mode, model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.
    test_f1 = 0.
    all_pred = []
    all_label = []

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    if(mode == 'transformer'):
        for batch_idx, (data_s, coord_s, data_l, coords_l, label) in enumerate(loader):
            data_s, coord_s, data_l, coords_l, label = data_s.to(device), coord_s.to(device), data_l.to(device), coords_l.to(device), label.to(device)
            slide_id = slide_ids.iloc[batch_idx]
            with torch.no_grad():
                if hasattr(model, 'config') or model.__class__.__name__ in {'ViLa_MIL_Model', 'HVTSurv'}:
                    _logits, Y_prob, _loss = model(data_s, coord_s, data_l, coords_l, label)
                    Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
                else:
                    K = min(data_s.size(0), data_l.size(0))
                    h = torch.cat([data_s[:K], data_l[:K]], dim=1)
                    _logits, Y_prob, Y_hat, _, _ = model(h)

            acc_logger.log(Y_hat, label)
            probs = Y_prob.cpu().numpy()
            all_probs[batch_idx] = probs
            all_labels[batch_idx] = label.item()

            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
            error = calculate_error(Y_hat, label)
            test_error += error

            all_pred.append(int(Y_hat.view(-1)[0].item()))
            all_label.append(int(label.view(-1)[0].item()))

        test_error /= len(loader)
        test_f1 = f1_score(all_label, all_pred, average='macro')

        if n_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
            aucs = []
        else:
            aucs = []
            binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
            for class_idx in range(n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))

            auc = np.nanmean(np.array(aucs))

        return patient_results, test_error, auc, acc_logger, test_f1

