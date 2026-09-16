from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import tv_tensors
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_annotations(json_path: str | Path, img_dir: str | Path):
    """Parse the row/column JSON into XYXY boxes, dropping anything unusable.

    Returns (annotations, class_to_idx). Label 0 is reserved for background, so the
    first real category is 1.
    """
    json_path, img_dir = Path(json_path), Path(img_dir)
    with open(json_path, "r") as f:
        records = json.load(f)

    categories = sorted({obj["category"] for rec in records for obj in rec["objects"]})
    class_to_idx = {name: i + 1 for i, name in enumerate(categories)}

    annotations = []
    stats = Counter()
    class_hist = Counter()

    for image_id, rec in enumerate(tqdm(records, desc="Parsing annotations", leave=False)):
        img_name = os.path.basename(rec["image"]["pathname"])
        if not (img_dir / img_name).exists():
            stats["images_missing_file"] += 1
            continue

        shape = rec["image"].get("shape", {})
        height, width = shape.get("r"), shape.get("c")

        boxes, labels = [], []
        for obj in rec["objects"]:
            bbox = obj["bounding_box"]
            xmin, ymin = float(bbox["minimum"]["c"]), float(bbox["minimum"]["r"])
            xmax, ymax = float(bbox["maximum"]["c"]), float(bbox["maximum"]["r"])

            # A handful of annotations in this dataset spill past the image border;
            # clamp first, then drop anything that is still degenerate.
            if width and height:
                xmin, xmax = max(0.0, xmin), min(float(width), xmax)
                ymin, ymax = max(0.0, ymin), min(float(height), ymax)
                stats["boxes_clamped"] += int(
                    bbox["minimum"]["c"] < 0
                    or bbox["minimum"]["r"] < 0
                    or bbox["maximum"]["c"] > width
                    or bbox["maximum"]["r"] > height
                )

            if xmax - xmin < 1.0 or ymax - ymin < 1.0:
                stats["boxes_degenerate"] += 1
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_to_idx[obj["category"]])
            class_hist[obj["category"]] += 1

        if not boxes:
            stats["images_without_boxes"] += 1
            continue

        annotations.append(
            {
                "path": img_name,
                "boxes": boxes,
                "labels": labels,
                "image_id": image_id,
                "height": height,
                "width": width,
            }
        )
        stats["images_kept"] += 1
        stats["boxes_kept"] += len(boxes)

    print(f"\nClasses ({len(class_to_idx)}): {class_to_idx}")
    print("Parse summary:", dict(stats))
    if class_hist:
        total = sum(class_hist.values())
        print("Box counts per class (this dataset is heavily imbalanced):")
        for name, count in class_hist.most_common():
            print(f"  {name:20s} {count:7d}  ({100 * count / total:5.2f}%)")
    return annotations, class_to_idx


class MalariaDataset(Dataset):
    # Returns (image, target) with boxes wrapped as tv_tensors.

    def __init__(self, img_dir, annotations, transforms=None):
        self.img_dir = Path(img_dir)
        self.annotations = annotations
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        ann = self.annotations[idx]
        img_path = self.img_dir / ann["path"]

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            # Fail loudly: a silent zero-image placeholder poisons training.
            raise RuntimeError(f"Could not read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        img = tv_tensors.Image(torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1))
        canvas_size = tuple(img.shape[-2:])  # (H, W)

        boxes = tv_tensors.BoundingBoxes(
            torch.as_tensor(ann["boxes"], dtype=torch.float32),
            format="XYXY",
            canvas_size=canvas_size,
        )
        target = {
            "boxes": boxes,
            "labels": torch.as_tensor(ann["labels"], dtype=torch.int64),
            "image_id": torch.tensor(ann["image_id"], dtype=torch.int64),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # Recomputed after transforms rather than carried over from the JSON.
        boxes_t = target["boxes"]
        target["area"] = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        target["iscrowd"] = torch.zeros((boxes_t.shape[0],), dtype=torch.int64)
        return img, target


def get_transform(train: bool):
    transforms = []
    if train:
        # Blood smears have no canonical orientation, so both flips are label-safe.
        transforms += [T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5)]
    transforms += [T.ToDtype(torch.float32, scale=True)]
    if train:
        transforms += [
            T.RandomPhotometricDistort(p=0.5),  # stain/illumination variation
            T.ClampBoundingBoxes(),
            T.SanitizeBoundingBoxes(),  # drops boxes (and their labels) left invalid
        ]
    transforms += [T.ToPureTensor()]
    return T.Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------- model


def build_model(num_classes: int, pretrained: bool = True, min_size: int = 800,
                max_size: int = 1333, detections_per_img: int = 300):
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=weights,
        min_size=min_size,
        max_size=max_size,
        trainable_backbone_layers=3 if pretrained else None,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # A single smear can hold 100+ red blood cells; the 100-detection default
    # silently caps recall, so raise it.
    model.roi_heads.detections_per_img = detections_per_img
    return model

# ------------------------------------------------------------------------- training

def train_one_epoch(model, dataloader, optimizer, device, epoch, scaler=None,
                    warmup_iters: int = 0, clip_grad: float = 10.0):
    model.train()
    running = 0.0

    warmup = None
    if epoch == 0 and warmup_iters > 0:
        # Detection losses are unstable in the first few hundred steps at lr=5e-3.
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, total_iters=warmup_iters
        )

    bar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [train]")
    for step, (imgs, targets) in enumerate(bar):
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step}: "
                f"{ {k: v.detach().item() for k, v in loss_dict.items()} }"
            )

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        if warmup is not None:
            warmup.step()

        loss_value = loss.detach().item()
        running += loss_value
        bar.set_postfix(loss=f"{loss_value:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    return running / max(len(dataloader), 1)


@torch.inference_mode()
def evaluate(model, dataloader, device, idx_to_class=None):
    # COCO-style mAP. The metric runs on CPU so it never competes for GPU memory.
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)

    for imgs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
        imgs = [img.to(device) for img in imgs]
        outputs = model(imgs)

        # torchmetrics only wants boxes/scores/labels; extra keys are dropped.
        preds = [
            {k: v.detach().cpu() for k, v in out.items() if k in ("boxes", "scores", "labels")}
            for out in outputs
        ]
        gts = [
            {k: v.cpu() for k, v in t.items() if k in ("boxes", "labels")}
            for t in targets
        ]
        metric.update(preds, gts)

    results = metric.compute()
    print(
        f"  mAP@[.5:.95] {float(results['map']):.4f} | "
        f"mAP@0.5 {float(results['map_50']):.4f} | "
        f"mAP small {float(results['map_small']):.4f} | "
        f"mAR@100 {float(results['mar_100']):.4f}"
    )
    if idx_to_class is not None and results.get("map_per_class") is not None:
        per_class = np.atleast_1d(results["map_per_class"].cpu().numpy())
        classes = np.atleast_1d(results["classes"].cpu().numpy())
        print("  per-class mAP:")
        for cls, value in zip(classes, per_class):
            print(f"    {idx_to_class.get(int(cls), int(cls)):20s} {float(value):.4f}")
    return float(results["map"]), float(results["map_50"])

# ------------------------------------------------------------------- synthetic data

def make_synthetic_dataset(root: Path, n_images: int = 6, size=(128, 160), seed: int = 0):
    # Tiny fake dataset in the real JSON schema, for pipeline tests only.
    rng = np.random.default_rng(seed)
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    height, width = size
    categories = ["red blood cell", "ring", "trophozoite"]

    records = []
    for i in range(n_images):
        img = rng.integers(150, 210, size=(height, width, 3), dtype=np.uint8)
        objects = []
        for _ in range(rng.integers(2, 5)):
            box_w, box_h = int(rng.integers(14, 26)), int(rng.integers(14, 26))
            x = int(rng.integers(0, width - box_w))
            y = int(rng.integers(0, height - box_h))
            category = categories[int(rng.integers(0, len(categories)))]
            colour = (60, 60, 200) if category == "red blood cell" else (40, 160, 60)
            cv2.circle(img, (x + box_w // 2, y + box_h // 2), min(box_w, box_h) // 2, colour, -1)
            objects.append(
                {
                    "category": category,
                    "bounding_box": {
                        "minimum": {"r": y, "c": x},
                        "maximum": {"r": y + box_h, "c": x + box_w},
                    },
                }
            )
        name = f"synthetic_{i:03d}.png"
        cv2.imwrite(str(img_dir / name), img)
        records.append(
            {
                "image": {"pathname": f"/images/{name}", "shape": {"r": height, "c": width}},
                "objects": objects,
            }
        )

    json_path = root / "training.json"
    with open(json_path, "w") as f:
        json.dump(records, f)
    return img_dir, json_path

# ----------------------------------------------------------------------------- main

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Faster R-CNN malaria parasite detection")
    p.add_argument("--img-dir", type=str, default=None)
    p.add_argument("--json", type=str, default=None)
    p.add_argument("--out", type=str, default="runs/frcnn")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-size", type=int, default=800)
    p.add_argument("--max-size", type=int, default=1333)
    p.add_argument("--limit", type=int, default=None, help="cap images, for quick runs")
    p.add_argument("--no-pretrained", action="store_true", help="skip the COCO weight download")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--smoke-test", action="store_true",
                   help="generate a tiny synthetic dataset and run one epoch")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    tmp_root = None
    if args.smoke_test:
        tmp_root = Path(tempfile.mkdtemp(prefix="malaria_smoke_"))
        img_dir, json_path = make_synthetic_dataset(tmp_root, n_images=6, size=(128, 160))
        args.img_dir, args.json = str(img_dir), str(json_path)
        args.epochs, args.batch_size, args.workers = 1, 2, 0
        args.min_size, args.max_size = 128, 160
        args.no_pretrained, args.no_amp = True, True
        args.out = str(tmp_root / "run")
        print(f"[smoke-test] synthetic dataset at {tmp_root}")

    if not args.img_dir or not args.json:
        print("Provide --img-dir and --json (or run with --smoke-test).", file=sys.stderr)
        return 2
    if not Path(args.img_dir).is_dir():
        print(f"--img-dir not found: {args.img_dir}", file=sys.stderr)
        return 2
    if not Path(args.json).is_file():
        print(f"--json not found: {args.json}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    annotations, class_to_idx = load_annotations(args.json, args.img_dir)
    if not annotations:
        print("No usable annotations. Check --img-dir and --json.", file=sys.stderr)
        return 1
    if args.limit:
        annotations = annotations[: args.limit]

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx) + 1  # + background
    print(f"Images: {len(annotations)} | classes incl. background: {num_classes}")

    # Split by image, with a fixed seed so the val set is stable across runs.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(annotations))
    n_val = max(1, int(round(len(annotations) * args.val_split)))
    val_idx, train_idx = order[:n_val], order[n_val:]
    if len(train_idx) == 0:  # degenerate only for toy datasets
        train_idx = val_idx
    train_anns = [annotations[i] for i in train_idx]
    val_anns = [annotations[i] for i in val_idx]
    print(f"Train images: {len(train_anns)} | val images: {len(val_anns)}")

    loader_train = DataLoader(
        MalariaDataset(args.img_dir, train_anns, get_transform(train=True)),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=collate_fn, pin_memory=(device.type == "cuda"),
        persistent_workers=args.workers > 0, drop_last=False,
    )
    loader_val = DataLoader(
        MalariaDataset(args.img_dir, val_anns, get_transform(train=False)),
        batch_size=1, shuffle=False, num_workers=args.workers,
        collate_fn=collate_fn, pin_memory=(device.type == "cuda"),
        persistent_workers=args.workers > 0,
    )

    model = build_model(
        num_classes,
        pretrained=not args.no_pretrained,
        min_size=args.min_size,
        max_size=args.max_size,
    ).to(device)

    start_epoch, best_map = 0, 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0)
        best_map = ckpt.get("best_map", 0.0)
        print(f"Resumed {args.resume} (epoch {start_epoch}, best mAP {best_map:.4f})")

    if args.eval_only:
        evaluate(model, loader_val, device, idx_to_class)
        if tmp_root:
            shutil.rmtree(tmp_root, ignore_errors=True)
        return 0

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda" and not args.no_amp
        else None
    )
    warmup_iters = min(500, max(len(loader_train) - 1, 0))

    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_one_epoch(
            model, loader_train, optimizer, device, epoch,
            scaler=scaler, warmup_iters=warmup_iters,
        )
        scheduler.step()
        print(f"Epoch {epoch + 1}/{args.epochs} | train loss {avg_loss:.4f}")

        val_map, val_map50 = evaluate(model, loader_val, device, idx_to_class)
        checkpoint = {
            "model": model.state_dict(),
            "class_to_idx": class_to_idx,
            "epoch": epoch + 1,
            "map": val_map,
            "map_50": val_map50,
            "best_map": max(best_map, val_map),
            "args": vars(args),
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if val_map > best_map:
            best_map = val_map
            torch.save(checkpoint, out_dir / "best.pt")
            print(f"  new best mAP {best_map:.4f} -> {out_dir / 'best.pt'}")

    print(f"\nFinished. Best validation mAP: {best_map:.4f}")
    if args.smoke_test:
        print("[smoke-test] pipeline ran end to end (metrics are meaningless on fake data)")
    if tmp_root:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
