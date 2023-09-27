import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
import random
from torch.utils.data.sampler import SubsetRandomSampler 
from ..model import vision_transformer as vits
from omegaconf import DictConfig
from collections import defaultdict
from typing import List, Tuple, Dict, Union, Sequence
import os
from .augmentations_ import RandomMixup, RandAugment, FixedAnglesRotateTransform
# Load the AGM dataset from the Hugging Face Hub
from datasets import load_dataset, load_from_disk
from datasets import ClassLabel

def unnormalize(img, mean, std):
    img = img * torch.Tensor(std).unsqueeze(1).unsqueeze(1) + torch.Tensor(mean).unsqueeze(1).unsqueeze(1)  
    return img

def get_model(cfg: DictConfig, classes):
    if cfg.arch == "vit_small" or cfg.arch == "vit_base":
        model = vits.__dict__[cfg.arch](img_size=cfg.img_size, patch_size=cfg.patch_size, num_classes=len(classes))
        model_head = vits.MLP(model.embed_dim, len(classes), cfg.mlp_num_layers, cfg.mlp_hidden_size)    
        model = torch.nn.Sequential(model, model_head)
    elif cfg.arch == "resnet":
        # load resnet model
        model = resnet50(pretrained=False)
        model.fc = vits.MLP(model.fc.in_features, len(classes), cfg.mlp_num_layers, cfg.mlp_hidden_size)   
        # model.fc = nn.Linear(model.fc.in_features, len(classes))
    # from timm.models.vision_transformer import VisionTransformer
    # model = VisionTransformer(
    #     img_size=cfg.img_size, patch_size=cfg.patch_size, in_chans=3, 
    #     num_classes=0, embed_dim=384, depth=12,
    #     num_heads=6, mlp_ratio=4., qkv_bias=True,
    #     norm_layer=partial(nn.LayerNorm, eps=1e-6),
    # )
    return model

def load_from_url_or_disk(remote_repo, local_repo, save_to_local):
    if local_repo:
        try:
            ds = load_from_disk(local_repo)
        except FileNotFoundError:
            ds = load_dataset(remote_repo)
    else:
        ds = load_dataset(remote_repo)
        
    if save_to_local:
        ds.save_to_disk(local_repo)
    return ds

def split_dataset(cfg, ds, collate_fn):
    # Generate splits
    if cfg.stratify:
        stratify = "label"
    else:
        stratify = None
    if cfg.test_split > 0:
        splits = ds["train"].train_test_split(test_size=0.2, seed=42, stratify_by_column=stratify)
        train_ds, test_ds = splits["train"], splits["test"]
        test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    else:
        train_ds = ds['train']
        test_ds = None
        test_dl = None
    if cfg.validation_split > 0:
        splits = train_ds.train_test_split(test_size=0.2, seed=42, stratify_by_column=stratify)
        train_ds, val_ds = splits["train"], splits["test"]
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
        val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    else:
        train_ds = ds['train']
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
        val_ds = None
        val_dl = None

    return train_ds, val_ds, test_ds, train_dl, val_dl, test_dl

def get_sets(cfg: DictConfig):

    # Set up augmentations
    if cfg.transforms == "default":
        transform = transforms.Compose([
            transforms.Resize(cfg.img_size, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(cfg.ds_mean, cfg.ds_std)
        ])
    elif cfg.transforms == "rot_and_flip":
        transform = transforms.Compose([
            transforms.Resize(cfg.img_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(90),
            FixedAnglesRotateTransform([0, 90, 180, 270]),
            transforms.ToTensor(),
            transforms.Normalize(cfg.ds_mean, cfg.ds_std)
        ])
    elif cfg.transforms == "full":
        from torch.utils.data.dataloader import default_collate
        transform = transforms.Compose([
            transforms.Resize(cfg.img_size, interpolation=Image.BICUBIC),
            RandAugment(cfg.randaugment_number, cfg.randaugment_magnitude),
            transforms.ToTensor(),
            transforms.Normalize(cfg.ds_mean, cfg.ds_std),
        ])
        num_classes = len([f for f in os.listdir(cfg.ds_dir) if os.path.isdir(os.path.join(cfg.ds_dir, f))])
        mixup_transform = RandomMixup(num_classes=num_classes, p=cfg.mixup_prob, alpha=cfg.mixup_alpha)
        def collate_fn(batch):
            return mixup_transform(*default_collate(batch))
        # pass
    else:
        raise ValueError(f"Transforms {cfg.transforms} not supported.")
    
    # Augmentations mapping function
    def transform_ds(sample):
        sample['image'] = [transform(x) for x in sample['image']]
        return sample

    # Load dataset
    ds = load_from_url_or_disk(cfg.ds_remote_repo, cfg.ds_local_repo, cfg.ds_save_to_local)

    class_names = sorted(ds['train'].unique("label"))
    class_to_idx = ClassLabel(names=class_names)._str2int
    classes = list(class_to_idx.keys())
    train_labels = ds["train"]["label"]
    samples_per_class = np.unique(train_labels, return_counts=True)
    class_weights =  1 - samples_per_class[1] / np.sum(samples_per_class[1])

    ds['train'] = ds['train'].class_encode_column("label")
    ds = ds.with_transform(transform_ds)

    train_ds, val_ds, test_ds, train_dl, val_dl, test_dl = split_dataset(cfg, ds, collate_fn if cfg.transforms == "full" else None)

    return train_ds, val_ds, test_ds, train_dl, val_dl, test_dl, classes, class_to_idx, class_weights

def merge_classes(x: torch.Tensor, merge_classes: List[List[int]]):
    # merge classes
    for classes in merge_classes:
        if x[1] == classes[0]:
            x[1] = classes[1]
    return x


def ft_sets(cfg: DictConfig):

    if cfg.merge_classes:
        # merge_class_dict = {}
        # for classes_to_merge in cfg.merge_classes:
        #     assert len(classes_to_merge) > 1, "You need to merge at least 2 classes"
        #     assert len(classes_to_merge) == len(set(classes_to_merge)), "You cannot merge the same class twice"
        transform = transforms.Compose([
            transforms.Resize(cfg.img_size, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(cfg.ds_mean, cfg.ds_std),
            transforms.Lambda(lambda x: merge_classes(x, cfg.merge_classes))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(cfg.img_size, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(cfg.ds_mean, cfg.ds_std)
        ])
    ds = ImageFolder(cfg.ds_dir, transform=transform)
    classes = ds.classes
    samples = ds.samples
    class_to_idx = ds.class_to_idx
    samples_per_class = np.unique(ds.targets, return_counts=True)
    # normalize
    class_weights =  1 - samples_per_class[1] / np.sum(samples_per_class[1])
       
    if cfg.test_split > 0:
        n_test = int(len(ds) * cfg.test_split)
        test_ds, ds = random_split(ds, [n_test, len(ds) - n_test])
        test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=True)
    else:
        test_ds = None
        test_dl = None
    if cfg.validation_split > 0:
        n_val = int(len(ds) * cfg.validation_split)
        train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=True)
    else:
        train_ds = ds
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_ds = None
        val_dl = None

    return train_ds, val_ds, test_ds, train_dl, val_dl, test_dl, classes, samples, class_to_idx, class_weights
def calculate_metrics(preds, labels):
    correct = torch.sum(preds == labels).item()
    precision = correct / len(preds)
    recall = correct / len(labels)
    f1 = 2 * (precision * recall) / (precision + recall)
    return correct, precision, recall, f1

def train_or_validate(model, dataloader, criterion, optimizer, device, is_training=True):
    if is_training:
        model.train()
        mode = "Train"
    else:
        model.eval()
        mode = "Validation"

    running_loss = 0.0
    running_corrects = 0
    running_precision = 0.0
    running_recall = 0.0
    all_labels = []
    all_preds = []

    for i, data in tqdm(enumerate(dataloader)):
        inputs, labels = data['image'], data['label']
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_training):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    correct, precision, recall, f1 = calculate_metrics(torch.tensor(all_preds), torch.tensor(all_labels))

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_corrects / len(dataloader.dataset)
    epoch_precision = precision
    epoch_recall = recall
    epoch_f1 = f1

    print(f"{mode} Loss: {epoch_loss:.3f}, Acc: {epoch_acc:.3f}, Precision: {epoch_precision:.3f}, Recall: {epoch_recall:.3f}, F1: {epoch_f1:.3f}")

    return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1, all_labels, all_preds

