"""
FedJD v12: Federated Jacobian Descent (torchjd-optional backend)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import parameters_to_vector
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import os
import random
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import warnings

try:
    from torchjd.autojac import jac_to_grad, mtl_backward
    from torchjd.aggregation import UPGrad, UPGradWeighting
    from torchjd.autogram import Engine
    TORCHJD_AVAILABLE = True
except Exception:
    jac_to_grad = None
    mtl_backward = None
    UPGrad = None
    UPGradWeighting = None
    Engine = None
    TORCHJD_AVAILABLE = False

warnings.filterwarnings('ignore')

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


@dataclass
class Hyperparams:
    clip_norm: float = 1.0
    patience: int = 10
    compress_ratio: float = 0.01
    quantize_bits: int = 8


HP = Hyperparams()


@dataclass
class Config:
    exp_id: str = ""
    exp_group: str = ""
    num_clients: int = 10
    num_rounds: int = 20
    local_epochs: int = 2
    batch_size: int = 128
    lr: float = 0.005
    local_moo_backend: str = "auto"  # auto | torchjd | native
    alpha: float = 0.5
    num_tasks: int = 5
    classes_per_task: int = 2
    ssjd_k: int = 3
    eval_freq: int = 5
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compress: bool = True
    workers: int = 0
    server_moo_beta: float = 0.5
    server_moo_mode: str = "loss_grad"  # sample_only | loss_only | grad_only | loss_grad
    server_solver: str = "heuristic"  # heuristic | simplex_qp
    server_qp_steps: int = 20
    server_qp_lr: float = 0.1
    server_fair_lambda: float = 0.1
    sketch_dim: int = 3
    server_sketch_gamma: float = 0.05
    client_fraction: float = 1.0
    loss_stat_batches: int = 1
    grad_stat_batches: int = 1
    grad_stat_task_limit: int = 0  # 0 means all valid tasks
    stat_every_rounds: int = 1
    parallel_clients: bool = True
    parallel_client_workers: int = 2
    prefetch_to_device: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True
    cudnn_benchmark: bool = True
    save_records: bool = True
    records_path: str = "results/records.csv"
    raw_dir: str = "results/raw"
    run_tag: str = ""
    notes: str = ""
    data_root: str = "./data"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader_kwargs(cfg: Config, shuffle: bool) -> Dict:
    kwargs = {
        "batch_size": cfg.batch_size,
        "shuffle": shuffle,
        "num_workers": cfg.workers,
    }
    if cfg.workers > 0:
        kwargs["pin_memory"] = cfg.pin_memory
        kwargs["persistent_workers"] = cfg.persistent_workers
    return kwargs


def iter_device_batches(loader: DataLoader, device: str, prefetch: bool = True):
    use_cuda = str(device).startswith("cuda")
    if not (prefetch and use_cuda):
        for imgs, labs, raw in loader:
            imgs = imgs.to(device, non_blocking=use_cuda)
            labs = [l.to(device, non_blocking=use_cuda) for l in labs]
            yield imgs, labs, raw
        return

    stream = torch.cuda.Stream(device=device)
    loader_it = iter(loader)

    def _prefetch(batch):
        imgs, labs, raw = batch
        with torch.cuda.stream(stream):
            imgs = imgs.to(device, non_blocking=True)
            labs = [l.to(device, non_blocking=True) for l in labs]
        return imgs, labs, raw

    try:
        next_batch = _prefetch(next(loader_it))
    except StopIteration:
        return

    while True:
        torch.cuda.current_stream(device=device).wait_stream(stream)
        imgs, labs, raw = next_batch
        try:
            next_batch = _prefetch(next(loader_it))
            has_next = True
        except StopIteration:
            has_next = False
        yield imgs, labs, raw
        if not has_next:
            break


def append_record_row(
    cfg: Config,
    method_name: str,
    hist: "History",
    elapsed_sec: float,
    run_id: Optional[str] = None
) -> None:
    if not cfg.save_records:
        return

    out_path = cfg.records_path
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fields = [
        "exp_id", "run_id", "exp_group", "alpha", "method", "local_moo_backend", "server_moo_mode", "server_moo_beta",
        "server_solver", "server_qp_steps", "server_qp_lr", "server_fair_lambda", "sketch_dim",
        "server_sketch_gamma",
        "loss_stat_batches", "grad_stat_batches", "stat_every_rounds", "seed", "num_rounds",
        "avg_acc", "worst_task_acc", "fairness", "elapsed_sec", "notes"
    ]
    row = {
        "exp_id": cfg.exp_id,
        "run_id": run_id or cfg.run_tag or f"{method_name}_seed{cfg.seed}_a{cfg.alpha}",
        "exp_group": cfg.exp_group,
        "alpha": cfg.alpha,
        "method": method_name,
        "local_moo_backend": cfg.local_moo_backend,
        "server_moo_mode": cfg.server_moo_mode,
        "server_moo_beta": cfg.server_moo_beta,
        "server_solver": cfg.server_solver,
        "server_qp_steps": cfg.server_qp_steps,
        "server_qp_lr": cfg.server_qp_lr,
        "server_fair_lambda": cfg.server_fair_lambda,
        "sketch_dim": cfg.sketch_dim,
        "server_sketch_gamma": cfg.server_sketch_gamma,
        "loss_stat_batches": cfg.loss_stat_batches,
        "grad_stat_batches": cfg.grad_stat_batches,
        "stat_every_rounds": cfg.stat_every_rounds,
        "seed": cfg.seed,
        "num_rounds": cfg.num_rounds,
        "avg_acc": float(hist.accs[-1]) if hist.accs else "",
        "worst_task_acc": float(min(hist.task_accs[-1])) if hist.task_accs else "",
        "fairness": float(hist.fairness[-1]) if hist.fairness else "",
        "elapsed_sec": float(elapsed_sec),
        "notes": cfg.notes,
    }
    write_header = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)


def write_round_log(cfg: Config, run_id: str, method_name: str, hist: "History") -> None:
    out_dir = os.path.join(cfg.raw_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{method_name}_round_metrics.csv")

    fields = ["round", "avg_acc", "fairness", "task_accs"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rnd, avg_acc, fair, task_acc in zip(hist.rounds, hist.accs, hist.fairness, hist.task_accs):
            w.writerow({
                "round": int(rnd),
                "avg_acc": float(avg_acc),
                "fairness": float(fair),
                "task_accs": ";".join(f"{float(x):.6f}" for x in task_acc),
            })


def write_round_timing_log(
    cfg: Config,
    run_id: str,
    method_name: str,
    rows: List[Dict[str, float]]
) -> None:
    out_dir = os.path.join(cfg.raw_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{method_name}_round_timing.csv")
    fields = ["round", "client_sec", "aggregate_sec", "eval_sec", "total_sec"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


@dataclass
class History:
    rounds: List[int] = field(default_factory=list)
    accs: List[float] = field(default_factory=list)
    task_accs: List[List[float]] = field(default_factory=list)
    fairness: List[float] = field(default_factory=list)
    name: str = ""


class Compressor:
    def __init__(self, ratio: float = 0.01, bits: int = 8):
        self.ratio = ratio
        self.bits = bits

    def compress(self, grads: List[torch.Tensor]) -> Tuple[List, Dict]:
        compressed, meta = [], {"shape": [], "scale": [], "full": []}
        for g in grads:
            flat = g.flatten()
            k = max(1, int(flat.numel() * self.ratio))
            if k >= flat.numel():
                compressed.append(flat.clone())
                meta["full"].append(True)
                meta["scale"].append(1.0)
            else:
                _, idx = torch.topk(flat.abs(), k)
                vals, scale = self._quantize(flat[idx], k)
                compressed.append((idx, vals, k))
                meta["full"].append(False)
                meta["scale"].append(scale)
            meta["shape"].append(g.shape)
        return compressed, meta

    def decompress(self, compressed: List, meta: Dict) -> List[torch.Tensor]:
        out = []
        for i, (c, shape) in enumerate(zip(compressed, meta["shape"])):
            if meta["full"][i]:
                out.append(c.view(shape))
            else:
                idx, vals, k = c
                full = torch.zeros(shape.numel(), device=vals.device)
                full[idx] = self._dequantize(vals, meta["scale"][i])
                out.append(full.view(shape))
        return out

    def _quantize(self, t: torch.Tensor, k: int) -> Tuple[torch.Tensor, float]:
        if k == 0:
            return t, 1.0
        mx = t.abs().max()
        if mx == 0:
            return t, 1.0
        lvl = 2 ** self.bits - 1
        return torch.round(t / mx * lvl), (mx / lvl).item()

    def _dequantize(self, q: torch.Tensor, s: float) -> torch.Tensor:
        return q.float() * s


class MultiHeadNet(nn.Module):
    def __init__(self, n_tasks: int = 5, n_classes: int = 2):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        resnet.maxpool = nn.Identity()

        resnet.bn1 = nn.InstanceNorm2d(64, track_running_stats=False)
        for layer in [resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]:
            for block in layer:
                if hasattr(block, 'bn1'):
                    block.bn1 = nn.InstanceNorm2d(block.bn1.num_features, track_running_stats=False)
                if hasattr(block, 'bn2'):
                    block.bn2 = nn.InstanceNorm2d(block.bn2.num_features, track_running_stats=False)

        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            resnet.avgpool, nn.Flatten()
        )
        self.heads = nn.ModuleList([
            nn.Linear(resnet.fc.in_features, n_classes) for _ in range(n_tasks)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        f = self.backbone(x)
        return f, [h(f) for h in self.heads]


class MTDataset(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform=None,
        n_tasks: int = 5,
        n_classes: int = 2,
        download: bool = False
    ):
        self.ds = torchvision.datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
        self.n_tasks = n_tasks
        self.n_classes = n_classes
        self.map = self._build_map()
        # Cache raw class targets for fast split without triggering __getitem__/transforms.
        self.targets = np.asarray(self.ds.targets, dtype=np.int64)

    def _build_map(self) -> Dict[int, Tuple[int, int]]:
        return {c: (c // self.n_classes, c % self.n_classes) for c in range(10) if c // self.n_classes < self.n_tasks}

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, label = self.ds[idx]
        labels = [-1] * self.n_tasks
        if label in self.map:
            tid, lid = self.map[label]
            labels[tid] = lid
        return img, labels, label


def split_data(dataset: Dataset, n_clients: int, alpha: float) -> List[List[int]]:
    if hasattr(dataset, "targets"):
        labels = np.asarray(dataset.targets, dtype=np.int64)
    else:
        labels = np.array([dataset[i][2] for i in range(len(dataset))])
    dist = np.random.dirichlet([alpha] * n_clients, len(np.unique(labels)))
    indices = [[] for _ in range(n_clients)]
    for c in range(len(np.unique(labels))):
        c_idx = np.where(labels == c)[0]
        np.random.shuffle(c_idx)
        splits = (np.cumsum(dist[c][:-1]) * len(c_idx)).astype(int)
        for cid, idx in enumerate(np.split(c_idx, splits)):
            indices[cid].extend(idx.tolist())
    return indices


def is_cifar10_ready(root: str) -> bool:
    base = os.path.join(root, "cifar-10-batches-py")
    required = [
        "batches.meta",
        "test_batch",
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]
    return os.path.isdir(base) and all(os.path.isfile(os.path.join(base, x)) for x in required)


class Optimizer:
    def __init__(self, model: nn.Module, cfg: Config, method: str = "ssjd"):
        self.model = model
        self.cfg = cfg
        self.method = method
        self.params = list(model.parameters())
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        self.opt = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=1e-4)
        self.backend = self._resolve_backend()
        self.agg = UPGrad() if (self.backend == "torchjd" and UPGrad is not None) else None
        self.engine = Engine(model.backbone, batch_dim=0) if (self.backend == "torchjd" and method == "iwrm" and Engine is not None) else None
        self.weighting = UPGradWeighting() if (self.backend == "torchjd" and method == "iwrm" and UPGradWeighting is not None) else None

    def _resolve_backend(self) -> str:
        req = str(self.cfg.local_moo_backend).lower().strip()
        if req == "torchjd":
            if not TORCHJD_AVAILABLE:
                print("[Warn] local_moo_backend='torchjd' but torchjd is unavailable; fallback to native.")
                return "native"
            return "torchjd"
        if req == "native":
            return "native"
        return "torchjd" if TORCHJD_AVAILABLE else "native"

    def step(self, feats: torch.Tensor, logits: List[torch.Tensor], labels: List[torch.Tensor]) -> float:
        method = self.method
        return getattr(self, f"_step_{method}")(feats, logits, labels) if method else 0.0

    def _apply_update(self):
        torch.nn.utils.clip_grad_norm_(self.params, HP.clip_norm)
        self.opt.step()
        self.opt.zero_grad()

    def _step_ssjd(self, feats: torch.Tensor, logits: List[torch.Tensor], labels: List[torch.Tensor]) -> float:
        tasks = random.sample(range(self.cfg.num_tasks), min(self.cfg.ssjd_k, self.cfg.num_tasks))
        valid = [(t, labels[t] != -1) for t in tasks if (labels[t] != -1).sum() > 0]
        losses = [self.criterion(logits[t][m], labels[t][m]).mean() for t, m in valid]
        if not losses:
            return 0.0

        self.opt.zero_grad()
        if len(losses) == 1:
            losses[0].backward()
        else:
            if self.backend == "torchjd" and mtl_backward is not None and jac_to_grad is not None and self.agg is not None:
                # torchjd path: detached-head update + Jacobian aggregation for shared backbone.
                head_losses = [self.criterion(self.model.heads[t](feats.detach())[m], labels[t][m]).mean()
                               for t, m in valid]
                torch.stack(head_losses).mean().backward()
                mtl_backward(losses, features=feats)
                jac_to_grad(self.model.backbone.parameters(), self.agg)
            else:
                # Native fallback: simple averaged task loss update.
                torch.stack(losses).mean().backward()

        self._apply_update()
        return sum(losses).item() / len(losses)

    def _step_iwrm(self, feats: torch.Tensor, logits: List[torch.Tensor], labels: List[torch.Tensor]) -> float:
        mat = []
        for t in range(self.cfg.num_tasks):
            m = labels[t] != -1
            if m.sum() > 0:
                mat.append(self.criterion(logits[t][m], labels[t][m]))
        if not mat:
            return 0.0

        mx = max(l.size(0) for l in mat)
        padded = [torch.cat([l, torch.zeros(mx - l.size(0), device=l.device)]) if l.size(0) < mx else l[:mx] for l in mat]
        losses = torch.stack(padded, dim=1)

        self.opt.zero_grad()
        if self.backend == "torchjd" and self.engine is not None and self.weighting is not None:
            gramian = self.engine.compute_gramian(losses)
            weights = self.weighting(gramian)
            losses.backward(weights)
        else:
            # Native fallback: average instance-wise objective.
            losses.mean().backward()
        self._apply_update()
        return losses.mean().item()

    def _step_cagrad(self, feats: torch.Tensor, logits: List[torch.Tensor], labels: List[torch.Tensor]) -> float:
        grads, losses = [], []
        valid_tasks = [t for t in range(self.cfg.num_tasks) if (labels[t] != -1).sum() > 0]
        for idx, t in enumerate(valid_tasks):
            m = labels[t] != -1
            loss = self.criterion(logits[t][m], labels[t][m]).mean()
            losses.append(loss)
            grads_t = torch.autograd.grad(
                loss,
                self.params,
                retain_graph=(idx < len(valid_tasks) - 1),
                allow_unused=True
            )
            flat_grad = []
            for p, g in zip(self.params, grads_t):
                if g is None:
                    flat_grad.append(torch.zeros_like(p).flatten())
                else:
                    flat_grad.append(g.detach().flatten())
            grads.append(torch.cat(flat_grad))

        if not grads:
            return 0.0
        if len(grads) == 1:
            self.opt.zero_grad()
            off = 0
            only_grad = grads[0]
            for p in self.params:
                p.grad = only_grad[off:off+p.numel()].view_as(p).clone()
                off += p.numel()
            self._apply_update()
            return losses[0].item()

        avg_grad = torch.stack(grads).mean(dim=0)
        c = 0.5

        adjusted = []
        for g in grads:
            dot = torch.dot(g, avg_grad)
            if dot < -1e-8:
                proj = (dot / (avg_grad.norm()**2 + 1e-8)) * avg_grad
                adjusted.append(g - (1 + c) * proj)
            else:
                adjusted.append(g)

        final_grad = torch.stack(adjusted).mean(dim=0)

        self.opt.zero_grad()
        off = 0
        for p in self.params:
            p.grad = final_grad[off:off+p.numel()].view_as(p).clone()
            off += p.numel()

        self._apply_update()
        return sum(l.item() for l in losses) / len(losses)

    def _step_fedavg(self, feats: torch.Tensor, logits: List[torch.Tensor], labels: List[torch.Tensor]) -> float:
        total, cnt = 0.0, 0
        for t in range(self.cfg.num_tasks):
            m = labels[t] != -1
            if m.sum() > 0:
                total += self.criterion(logits[t][m], labels[t][m]).mean()
                cnt += 1
        if cnt == 0:
            return 0.0

        self.opt.zero_grad()
        (total / cnt).backward()
        self._apply_update()
        return (total / cnt).item()


class Evaluator:
    @staticmethod
    def compute(model: nn.Module, loader: DataLoader, cfg: Config, device: torch.device) -> Tuple[List[float], float]:
        model.eval()
        correct, total = [0] * cfg.num_tasks, [0] * cfg.num_tasks
        with torch.no_grad():
            for imgs, labs, _ in iter_device_batches(loader, str(device), prefetch=cfg.prefetch_to_device):
                _, out = model(imgs)
                for t in range(cfg.num_tasks):
                    m = labs[t] != -1
                    if m.sum() > 0:
                        correct[t] += (out[t][m].argmax(1) == labs[t][m]).sum().item()
                        total[t] += m.sum().item()
        accs = [100.0 * c / max(t, 1) for c, t in zip(correct, total)]
        return accs, 1.0 - np.std(accs) / 100.0


class Client:
    def __init__(self, model: nn.Module, loader: DataLoader, cfg: Config, cid: int, method: str):
        self.model = model
        self.loader = loader
        self.cfg = cfg
        self.cid = cid
        self.opt = Optimizer(model, cfg, method)
        self.stat_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self._cached_task_losses = [0.0] * self.cfg.num_tasks
        self._cached_task_grad_norms = [0.0] * self.cfg.num_tasks
        self._cached_task_sketch = [0.0] * max(1, int(self.cfg.sketch_dim))

    def get_losses(self, max_batches: Optional[int] = None) -> List[float]:
        self.model.eval()
        losses = [0.0] * self.cfg.num_tasks
        seen = 0
        with torch.no_grad():
            for imgs, labs, _ in iter_device_batches(self.loader, self.cfg.device, prefetch=self.cfg.prefetch_to_device):
                if max_batches is not None and seen >= max_batches:
                    break
                _, out = self.model(imgs)
                for t in range(self.cfg.num_tasks):
                    m = labs[t] != -1
                    if m.sum() > 0:
                        losses[t] += self.stat_criterion(out[t][m], labs[t][m]).item()
                seen += 1
        denom = max(seen, 1)
        return [l / denom for l in losses]

    def get_backbone_task_grad_norms(self, max_batches: int = 1) -> List[float]:
        self.model.train()
        norms = [0.0] * self.cfg.num_tasks
        cnt = [0] * self.cfg.num_tasks

        for bidx, (imgs, labs, _) in enumerate(iter_device_batches(self.loader, self.cfg.device, prefetch=self.cfg.prefetch_to_device)):
            if bidx >= max_batches:
                break
            _, out = self.model(imgs)

            valid_tasks = [t for t in range(self.cfg.num_tasks) if (labs[t] != -1).sum() > 0]
            t_limit = int(self.cfg.grad_stat_task_limit)
            if t_limit > 0 and len(valid_tasks) > t_limit:
                valid_tasks = random.sample(valid_tasks, t_limit)

            for t in valid_tasks:
                m = labs[t] != -1
                self.model.zero_grad()
                loss = self.stat_criterion(out[t][m], labs[t][m])
                loss.backward(retain_graph=(t != valid_tasks[-1]))
                sq = 0.0
                for p in self.model.backbone.parameters():
                    if p.grad is not None:
                        sq += p.grad.detach().pow(2).sum().item()
                norms[t] += float(np.sqrt(sq))
                cnt[t] += 1

        self.model.zero_grad()
        for t in range(self.cfg.num_tasks):
            if cnt[t] > 0:
                norms[t] /= cnt[t]
        return norms

    def train(
        self,
        comp: Optional[Compressor] = None,
        need_loss_stats: bool = True,
        need_grad_stats: bool = True,
        refresh_stats: bool = True
    ) -> Tuple[torch.Tensor, List[float], List[float], List[float], int, Optional[Dict]]:
        self.model.train()
        init = parameters_to_vector(self.model.parameters()).detach().clone()

        n_batch = 0
        for _ in range(self.cfg.local_epochs):
            for imgs, labs, _ in iter_device_batches(self.loader, self.cfg.device, prefetch=self.cfg.prefetch_to_device):
                feats, out = self.model(imgs)
                loss = self.opt.step(feats, out, labs)
                if loss > 0:
                    n_batch += 1

        delta = parameters_to_vector(self.model.parameters()).detach() - init
        if need_loss_stats:
            if refresh_stats:
                n_batches = max(1, int(self.cfg.loss_stat_batches))
                self._cached_task_losses = self.get_losses(max_batches=n_batches)
            task_losses = self._cached_task_losses
        else:
            task_losses = [0.0] * self.cfg.num_tasks

        if need_grad_stats:
            if refresh_stats:
                n_batches = max(1, int(self.cfg.grad_stat_batches))
                self._cached_task_grad_norms = self.get_backbone_task_grad_norms(max_batches=n_batches)
            task_grad_norms = self._cached_task_grad_norms
        else:
            task_grad_norms = [0.0] * self.cfg.num_tasks

        if refresh_stats:
            self._cached_task_sketch = self._build_task_sketch(task_losses, task_grad_norms)
        task_sketch = self._cached_task_sketch

        if comp and self.cfg.compress:
            delta, meta = comp.compress([delta])
            return delta[0], task_losses, task_grad_norms, task_sketch, n_batch, meta
        return delta, task_losses, task_grad_norms, task_sketch, n_batch, None

    def _build_task_sketch(self, task_losses: List[float], task_grad_norms: List[float]) -> List[float]:
        # Low-dimensional client summary used by server-side simplex solver.
        src = task_grad_norms if any(v > 0 for v in task_grad_norms) else task_losses
        v = torch.tensor(src, dtype=torch.float32)
        m = max(1, int(self.cfg.sketch_dim))
        g = torch.Generator(device="cpu")
        g.manual_seed(self.cfg.seed + 10007 * (self.cid + 1))
        proj = torch.randn(v.numel(), m, generator=g) / np.sqrt(max(v.numel(), 1))
        return (v @ proj).tolist()

    def set_params(self, params: torch.Tensor):
        off = 0
        with torch.no_grad():
            for p in self.model.parameters():
                p.copy_(params[off:off+p.numel()].view_as(p))
                off += p.numel()


class Server:
    def __init__(self, model: nn.Module, cfg: Config):
        self.model = model
        self.cfg = cfg
        self._warned_invalid_mode = False

    def _resolve_mode(self) -> str:
        valid_modes = {"sample_only", "loss_only", "grad_only", "loss_grad"}
        mode = str(self.cfg.server_moo_mode).lower().strip()
        if mode not in valid_modes:
            if not self._warned_invalid_mode:
                print(f"[Warn] Invalid server_moo_mode='{self.cfg.server_moo_mode}', fallback to 'loss_grad'.")
                self._warned_invalid_mode = True
            return "loss_grad"
        return mode

    @staticmethod
    def _sanitize(values: List[float]) -> torch.Tensor:
        return torch.tensor(
            [float(v) if np.isfinite(v) and v > 0 else 0.0 for v in values],
            dtype=torch.float32
        )

    @staticmethod
    def _safe_normalize(x: torch.Tensor) -> torch.Tensor:
        s = x.sum()
        if s > 0 and torch.isfinite(s):
            return x / s
        return torch.zeros_like(x)

    def _base_client_weights(self, weights: List[int]) -> torch.Tensor:
        base = torch.tensor([float(x) for x in weights], dtype=torch.float32)
        norm = self._safe_normalize(base)
        if norm.sum() == 0:
            return torch.ones_like(base) / max(len(base), 1)
        return norm

    def _client_moo_scores(
        self,
        task_losses: List[List[float]],
        task_grad_norms: List[List[float]],
        mode: str
    ) -> torch.Tensor:
        scores = []
        for losses, norms in zip(task_losses, task_grad_norms):
            l = self._sanitize(losses)
            g = self._sanitize(norms)

            if mode == "loss_only":
                valid_l = l[l > 0]
                scores.append(float(valid_l.max().item()) if valid_l.numel() > 0 else 0.0)
                continue

            if mode == "grad_only":
                valid_g = g[g > 0]
                scores.append(float(valid_g.max().item()) if valid_g.numel() > 0 else 0.0)
                continue

            # Default "loss_grad": emphasize tasks with both high loss and high backbone sensitivity.
            ln = self._safe_normalize(l)
            gn = self._safe_normalize(g)
            scores.append(float((ln * gn).sum().item()))

        return torch.tensor(scores, dtype=torch.float32)

    def _mix_weights(self, base_w: torch.Tensor, moo_scores: torch.Tensor, beta: float) -> torch.Tensor:
        moo_w = self._safe_normalize(moo_scores)
        if moo_w.sum() == 0:
            return base_w
        return (1.0 - beta) * base_w + beta * moo_w

    @staticmethod
    def _project_to_simplex(v: torch.Tensor) -> torch.Tensor:
        # Euclidean projection onto simplex {x >= 0, sum x = 1}.
        if v.numel() == 1:
            return torch.ones_like(v)
        u, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(u, dim=0) - 1
        ind = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
        cond = u - cssv / ind > 0
        rho = torch.nonzero(cond, as_tuple=False)[-1, 0]
        theta = cssv[rho] / (rho + 1).to(v.dtype)
        return torch.clamp(v - theta, min=0.0)

    def _solve_client_weights_simplex_qp(
        self,
        updates: List[torch.Tensor],
        base_w: torch.Tensor,
        task_losses: Optional[List[List[float]]] = None,
        task_sketches: Optional[List[List[float]]] = None
    ) -> torch.Tensor:
        if len(updates) <= 1:
            return base_w

        U = torch.stack(updates).float()
        G = U @ U.t()
        alpha = base_w.clone().float()

        losses = None
        if task_losses:
            losses = torch.tensor(task_losses, dtype=torch.float32)
        sketches = None
        if task_sketches:
            sketches = torch.tensor(task_sketches, dtype=torch.float32)

        steps = max(1, int(self.cfg.server_qp_steps))
        lr = float(self.cfg.server_qp_lr)
        lam = float(self.cfg.server_fair_lambda)
        gamma = float(self.cfg.server_sketch_gamma)

        for _ in range(steps):
            grad = 2.0 * (G @ alpha)
            if losses is not None and losses.numel() > 0:
                task_obj = alpha @ losses
                t_star = int(torch.argmax(task_obj).item())
                grad = grad + lam * losses[:, t_star]
            if sketches is not None and sketches.numel() > 0:
                grad = grad + 2.0 * gamma * (sketches @ sketches.t() @ alpha)
            alpha = self._project_to_simplex(alpha - lr * grad)

        return self._safe_normalize(alpha)

    def select_clients(self, clients: List[Client]) -> List[int]:
        frac = float(np.clip(self.cfg.client_fraction, 0.0, 1.0))
        n = max(1, int(len(clients) * frac))
        return random.sample(range(len(clients)), n)

    def aggregate(self, updates: List[torch.Tensor], weights: List[int], method: str,
                  task_losses: Optional[List[List[float]]] = None,
                  task_grad_norms: Optional[List[List[float]]] = None,
                  task_sketches: Optional[List[List[float]]] = None) -> torch.Tensor:
        if not updates:
            return torch.zeros(1)

        base_w = self._base_client_weights(weights)
        beta = float(np.clip(self.cfg.server_moo_beta, 0.0, 1.0))
        mode = self._resolve_mode()
        solver = str(self.cfg.server_solver).lower().strip()

        if solver == "simplex_qp" and method in {"ssjd", "iwrm", "cagrad"}:
            qp_w = self._solve_client_weights_simplex_qp(updates, base_w, task_losses, task_sketches)
            w = (1.0 - beta) * base_w + beta * qp_w
            return sum(u * wi for u, wi in zip(updates, w))

        if mode == "sample_only":
            return sum(u * wi for u, wi in zip(updates, base_w))

        if method in {"ssjd", "iwrm", "cagrad"} and task_losses and task_grad_norms:
            moo_scores = self._client_moo_scores(task_losses, task_grad_norms, mode)
            w = self._mix_weights(base_w, moo_scores, beta)
        else:
            w = base_w

        return sum(u * wi for u, wi in zip(updates, w))

    def update(self, delta: torch.Tensor):
        off = 0
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(delta[off:off+p.numel()].view_as(p))
                off += p.numel()

    def get_params(self) -> torch.Tensor:
        return parameters_to_vector(self.model.parameters()).detach().clone()


def plot_results(histories: dict, save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'FedJD-SSJD': 'blue', 'FedJD-IWRM': 'green', 'CAGrad': 'orange', 'FedAvg': 'red'}

    for name, hist in histories.items():
        if hist.accs:
            axes[0].plot(hist.rounds, hist.accs, '-o', color=colors.get(name, 'gray'), label=name, lw=2, ms=6)

    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Test Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    final = {n: h.accs[-1] if h.accs else 0 for n, h in histories.items()}
    if final:
        axes[1].bar(range(len(final)), list(final.values()), color=[colors.get(n, 'gray') for n in final])
        axes[1].set_xticks(range(len(final)))
        axes[1].set_xticklabels(list(final.keys()), rotation=15)
        axes[1].set_ylabel('Accuracy (%)')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/results.png', dpi=150)
    plt.show()


def train_one_client_round(
    c: Client,
    params: torch.Tensor,
    comp: Optional[Compressor],
    need_loss_stats: bool,
    need_grad_stats: bool,
    refresh_stats: bool
):
    c.set_params(params)
    return c.train(
        comp,
        need_loss_stats=need_loss_stats,
        need_grad_stats=need_grad_stats,
        refresh_stats=refresh_stats
    )


def run(cfg: Config, methods: List[str] = None) -> dict:
    methods = methods or ["ssjd", "iwrm", "cagrad", "fedavg"]
    set_seed(cfg.seed)
    backend_req = str(cfg.local_moo_backend).lower().strip()
    backend_msg = "torchjd" if (backend_req == "auto" and TORCHJD_AVAILABLE) else backend_req
    print(f"[Backend] local_moo_backend={backend_msg} (torchjd_available={TORCHJD_AVAILABLE})")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    data_ready = is_cifar10_ready(cfg.data_root)
    if not data_ready:
        print(f"[Data] CIFAR-10 not ready at {cfg.data_root}, downloading/extracting once...")
    train_ds = MTDataset(
        cfg.data_root, True, transform, cfg.num_tasks, cfg.classes_per_task, download=(not data_ready)
    )
    test_ds = MTDataset(
        cfg.data_root, False, transform_test, cfg.num_tasks, cfg.classes_per_task, download=False
    )
    test_loader = DataLoader(test_ds, **make_loader_kwargs(cfg, shuffle=False))
    client_idx = split_data(train_ds, cfg.num_clients, cfg.alpha)
    comp = Compressor(HP.compress_ratio, HP.quantize_bits) if cfg.compress else None

    names = {"ssjd": "FedJD-SSJD", "iwrm": "FedJD-IWRM", "cagrad": "CAGrad", "fedavg": "FedAvg"}
    histories = {}

    for method in methods:
        print(f"\n{'='*50}\n{names.get(method, method.upper())}\n{'='*50}")
        try:
            method_start = time.perf_counter()
            run_id = cfg.run_tag or f"{method}_seed{cfg.seed}_a{cfg.alpha}"
            if len(methods) > 1 and cfg.run_tag:
                run_id = f"{cfg.run_tag}_{method}"
            global_model = MultiHeadNet(cfg.num_tasks, cfg.classes_per_task).to(cfg.device)
            server = Server(global_model, cfg)

            clients = []
            for cid in range(cfg.num_clients):
                m = MultiHeadNet(cfg.num_tasks, cfg.classes_per_task).to(cfg.device)
                m.load_state_dict(global_model.state_dict())
                loader = DataLoader(Subset(train_ds, client_idx[cid]), **make_loader_kwargs(cfg, shuffle=True))
                clients.append(Client(m, loader, cfg, cid, method))

            hist = History(name=names[method])
            best_acc, patience = 0.0, 0
            params = server.get_params()
            round_timing_rows = []

            for r in tqdm(range(cfg.num_rounds), desc="Round"):
                t_round = time.perf_counter()
                selected = [clients[i] for i in server.select_clients(clients)]
                updates, task_losses, task_grad_norms, task_sketches, weights = [], [], [], [], []
                mode = server._resolve_mode()
                solver = str(cfg.server_solver).lower().strip()
                use_moo_stats = method in {"ssjd", "iwrm", "cagrad"} and mode != "sample_only"
                if solver == "simplex_qp" and method in {"ssjd", "iwrm", "cagrad"}:
                    use_moo_stats = True
                need_loss_stats = use_moo_stats and mode in {"loss_only", "loss_grad"}
                need_grad_stats = use_moo_stats and mode in {"grad_only", "loss_grad"}
                if solver == "simplex_qp":
                    need_loss_stats = True
                stat_interval = max(1, int(cfg.stat_every_rounds))
                refresh_stats = (r % stat_interval == 0)

                t_client0 = time.perf_counter()
                if cfg.parallel_clients and len(selected) > 1:
                    max_workers = max(1, min(int(cfg.parallel_client_workers), len(selected)))
                    with ThreadPoolExecutor(max_workers=max_workers) as ex:
                        futures = [
                            ex.submit(
                                train_one_client_round,
                                c,
                                params,
                                comp,
                                need_loss_stats,
                                need_grad_stats,
                                refresh_stats
                            )
                            for c in selected
                        ]
                        results = [f.result() for f in futures]
                else:
                    results = [
                        train_one_client_round(
                            c,
                            params,
                            comp,
                            need_loss_stats,
                            need_grad_stats,
                            refresh_stats
                        )
                        for c in selected
                    ]

                for delta, t_loss, t_norm, t_sketch, n, meta in results:
                    if meta and comp:
                        delta = comp.decompress([delta], meta)[0]
                    updates.append(delta)
                    task_losses.append(t_loss)
                    task_grad_norms.append(t_norm)
                    task_sketches.append(t_sketch)
                    weights.append(n)
                client_sec = time.perf_counter() - t_client0

                t_agg0 = time.perf_counter()
                server.update(server.aggregate(updates, weights, method, task_losses, task_grad_norms, task_sketches))
                params = server.get_params()
                aggregate_sec = time.perf_counter() - t_agg0

                eval_sec = 0.0
                if (r + 1) % cfg.eval_freq == 0 or r == cfg.num_rounds - 1:
                    t_eval0 = time.perf_counter()
                    accs, fair = Evaluator.compute(global_model, test_loader, cfg, cfg.device)
                    hist.rounds.append(r + 1)
                    hist.accs.append(np.mean(accs))
                    hist.task_accs.append(accs)
                    hist.fairness.append(fair)
                    eval_sec = time.perf_counter() - t_eval0

                    curr = np.mean(accs)
                    print(f"\n--- Round {r+1}/{cfg.num_rounds} ---")
                    print(f"  Acc: {curr:.2f}%  Fairness: {fair:.4f}")

                    if curr > best_acc:
                        best_acc, patience = curr, 0
                    elif (patience := patience + 1) >= HP.patience:
                        print(f"Early stop at round {r+1}")
                        round_timing_rows.append({
                            "round": int(r + 1),
                            "client_sec": float(client_sec),
                            "aggregate_sec": float(aggregate_sec),
                            "eval_sec": float(eval_sec),
                            "total_sec": float(time.perf_counter() - t_round),
                        })
                        break

                round_timing_rows.append({
                    "round": int(r + 1),
                    "client_sec": float(client_sec),
                    "aggregate_sec": float(aggregate_sec),
                    "eval_sec": float(eval_sec),
                    "total_sec": float(time.perf_counter() - t_round),
                })

            histories[hist.name] = hist
            print(f"\n{hist.name} Final: {hist.accs[-1]:.2f}%")
            elapsed = time.perf_counter() - method_start
            write_round_log(cfg, run_id, method, hist)
            write_round_timing_log(cfg, run_id, method, round_timing_rows)
            append_record_row(cfg, method, hist, elapsed, run_id=run_id)
            print(f"{hist.name} Time: {elapsed/60:.1f} min")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            histories[names.get(method, method)] = History(name=names.get(method, method))

    return histories


def main():
    cfg = Config()
    # Quick pilot run settings for limited Kaggle CPU/GPU budget.
    cfg.num_rounds = 10
    cfg.client_fraction = 0.3
    cfg.eval_freq = 10
    cfg.stat_every_rounds = 5
    cfg.loss_stat_batches = 1
    cfg.grad_stat_batches = 1
    cfg.workers = 2
    cfg.run_tag = "pilot_ssjd"

    print(f"Device: {cfg.device}  Tasks: {cfg.num_tasks}  Compress: {cfg.compress}")

    histories = run(cfg, ["ssjd"])
    plot_results(histories)

    print("\n" + "="*50 + "\nFINAL\n" + "="*50)
    for name, hist in histories.items():
        if hist.accs:
            print(f"{name}: {hist.accs[-1]:.2f}%")


if __name__ == "__main__":
    main()
