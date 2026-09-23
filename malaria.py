import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import argparse
import json  # Added for parsing
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import kagglehub

class MalariaDataset(Dataset):
    def __init__(self, img_dir, annotations, transforms=None):
        self.img_dir = img_dir
        self.annotations = annotations
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.img_dir, ann["path"])
        
        try:
            img = cv2.imread(img_path)

            if img is None:
                raise IOError(f"Could not read image: {img_path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Convert HWC -> CHW
            img = torch.from_numpy(img).permute(2, 0, 1)

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = torch.zeros((3, 224, 224), dtype=torch.uint8)
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "image_id": torch.tensor([idx]),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros(0, dtype=torch.int64)
            }
            if self.transforms:
                img = T.ToDtype(torch.float, scale=True)(img)
                img = T.ToPureTensor()(img)
                return img, target
            return img, target

        boxes = torch.as_tensor(ann["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(ann["labels"], dtype=torch.int64)
        image_id = torch.tensor([ann["image_id"]])
        
        if boxes.shape[0] > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.zeros(0, dtype=torch.float32)
            
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomPhotometricDistort(p=0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training")
    
    for imgs, targets in progress_bar:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        progress_bar.set_postfix(loss=f"{losses.item():.4f}")
        
    return total_loss / len(dataloader)

def collate_fn(batch):
    return tuple(zip(*batch))

@torch.inference_mode()
def evaluate(model, dataloader, device):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy").to(device)
    
    progress_bar = tqdm(dataloader, desc="Evaluating")
    for imgs, targets in progress_bar:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        predictions = model(imgs)
        metric.update(predictions, targets)

    try:
        results = metric.compute()
        print(f"\nValidation mAP: {results['map']:.4f}, mAP_50: {results['map_50']:.4f}\n")
        return results['map']
    except Exception as e:
        print(f"\nCould not compute mAP: {e}\n")
        return 0.0

def load_annotations_from_json(json_path, img_dir):
    """
    Loads annotations from the Kaggle 'training.json' file.
    """
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    all_annotations = []
    
    all_categories = set()
    for item in data:
        for obj in item['objects']:
            all_categories.add(obj['category'])
            
    class_to_idx = {name: i + 1 for i, name in enumerate(sorted(list(all_categories)))}
    print(f"Found classes: {class_to_idx}")

    for idx, item in enumerate(tqdm(data, desc="Loading Annotations")):
        
        img_name = os.path.basename(item['image']['pathname'])
        full_img_path = os.path.join(img_dir, img_name)
        
        if not os.path.exists(full_img_path):
            continue

        boxes = []
        labels = []

        for obj in item['objects']:
            category = obj['category']
            
            bbox = obj['bounding_box']
            ymin = bbox['minimum']['r']
            xmin = bbox['minimum']['c']
            ymax = bbox['maximum']['r']
            xmax = bbox['maximum']['c']

            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_to_idx[category])

        if len(boxes) > 0:
            all_annotations.append({
                "path": img_name,
                "boxes": boxes,
                "labels": labels,
                "image_id": idx
            })

    return all_annotations, class_to_idx

def main():
    # Download/access dataset
    path = kagglehub.dataset_download("kmader/malaria-bounding-boxes")
    print("Dataset path:", path)

    IMG_DIR = os.path.join(path, "malaria", "images")
    JSON_PATH = os.path.join(path, "malaria", "training.json")
    
    BATCH_SIZE = 4
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.005
    
    if IMG_DIR == "path/to/malaria-bounding-boxes/malaria/images" or \
       JSON_PATH == "path/to/malaria-bounding-boxes/malaria/training.json":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: Please update IMG_DIR and JSON_PATH     !!!")
        print("!!! in the main() function before running.         !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_annotations, class_to_idx = load_annotations_from_json(JSON_PATH, IMG_DIR)
    
    NUM_CLASSES = len(class_to_idx) + 1 
    print(f"Total classes: {NUM_CLASSES} (including background)")
    print(f"Total images with annotations: {len(all_annotations)}")
    
    if len(all_annotations) == 0:
        print(f"Error: No annotations were loaded. Check your JSON_PATH and IMG_DIR.")
        return

    np.random.seed(42)
    np.random.shuffle(all_annotations)
    
    split_idx = int(len(all_annotations) * 0.8)
    
    if split_idx == 0 or split_idx == len(all_annotations):
         print("Warning: Dataset too small for a train/val split. Using all data for both.")
         train_annotations = all_annotations
         val_annotations = all_annotations
    else:
        train_annotations = all_annotations[:split_idx]
        val_annotations = all_annotations[split_idx:]
    
    print(f"Training samples: {len(train_annotations)}")
    print(f"Validation samples: {len(val_annotations)}")

    dataset_train = MalariaDataset(IMG_DIR, train_annotations, get_transform(train=True))
    dataset_val = MalariaDataset(IMG_DIR, val_annotations, get_transform(train=False))

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=1, 
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    model = get_model(NUM_CLASSES).to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_map = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        avg_loss = train_one_epoch(model, dataloader_train, optimizer, device)
        print(f"Epoch {epoch+1} Average Training Loss: {avg_loss:.4f}")
        
        lr_scheduler.step()
        
        if dataloader_val and len(dataloader_val) > 0:
            val_map = evaluate(model, dataloader_val, device)
            
            if val_map > best_map:
                best_map = val_map
                torch.save(model.state_dict(), "best_model.pth")
                print(f"New best model saved with mAP: {best_map:.4f}")

    print("--- Training Finished ---")
    print(f"Best validation mAP: {best_map:.4f}")


if __name__ == "__main__":
    main()
