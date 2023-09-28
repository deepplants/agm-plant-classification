import os
from typing import Dict, List, Tuple
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import timm
from torchvision import transforms
from torchvision.models import resnet50
from .train_util import load_from_url_or_disk, split_dataset
from .datasets_util import ImageFolderPlantDoc, CustomImageFolder
from ..model import vision_transformer as vits
from datasets import ClassLabel
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import collections

def split_ft_dataset(cfg, ds):
    # split dataset
    if cfg.test:
        val_split = int(len(ds)*cfg.validation_split)
        test_split = int(len(ds)*cfg.test_split)
        train_split = len(ds) - val_split - test_split
        train_ds, val_ds, test_ds = random_split(ds, [train_split, val_split, test_split])
    else:
        val_split = int(len(ds)*cfg.validation_split)
        train_split = len(ds) - val_split
        train_ds, val_ds = random_split(ds, [train_split, val_split])
    
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    if cfg.test:
        print(f"Test dataset size: {len(test_ds)}")
    
    # create dataloaders
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=True)
    if cfg.test:
        test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=True)
    else:
        test_ds = None
        test_dl = None

    print(f"Train iterations per epoch: {len(train_dl)}")
    print(f"Validation iterations per epoch: {len(val_dl)}")

    return train_ds, val_ds, test_ds, train_dl, val_dl, test_dl

def get_finetune_sets(cfg: Dict):

    transform = transforms.Compose([
        transforms.Resize(cfg.img_size, interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(cfg.ds_mean, cfg.ds_std)
    ])

    # Augmentations mapping function
    def transform_ds(sample):
        sample['image'] = [transform(x) for x in sample['image']]
        return sample
    
    # Augmentations mapping function
    def transform_ds_hs(sample):
        sample['image'] = [transform(x) for x in sample['image']]
        sample.pop('mask')
        return sample

    # load dataset
    if cfg.fine_tune_on == "agm_hs":          
        ds = load_from_url_or_disk(cfg.ds_remote_repo, cfg.ds_local_repo, cfg.ds_save_to_local)
        ds['train'] = ds['train'].class_encode_column("label")
        ds = ds.with_transform(transform_ds_hs)
        class_names = sorted(ds['train'].unique("label"))
        class_to_idx = ClassLabel(names=class_names)._str2int
        classes = list(class_to_idx.keys())
    elif cfg.fine_tune_on == "agm":
        ds = load_from_url_or_disk(cfg.ds_remote_repo, cfg.ds_local_repo, cfg.ds_save_to_local)
        ds['train'] = ds['train'].class_encode_column("label")
        ds = ds.with_transform(transform_ds)
        class_names = sorted(ds['train'].unique("label"))
        class_to_idx = ClassLabel(names=class_names)._str2int
        classes = list(class_to_idx.keys())
    elif cfg.fine_tune_on == "plant_doc":
        ds = ImageFolderPlantDoc(cfg.ds_local_repo, transform=transform)
        classes = ds.classes
        class_to_idx = ds.class_to_idx
    elif cfg.fine_tune_on == "rice_plant":
        ds = CustomImageFolder(cfg.ds_local_repo, transform=transform)
        classes = ds.classes
        class_to_idx = ds.class_to_idx
    elif cfg.fine_tune_on == "cassava":
        ds = CustomImageFolder(cfg.ds_local_repo, transform=transform)
        classes = ds.classes
        class_to_idx = ds.class_to_idx
    else:
        raise ValueError(f"Fine tune dataset {cfg.fine_tune_on} not supported.")
    
    if cfg.fine_tune_on in ["agm", "agm_hs"]:
        train_ds, val_ds, test_ds, train_dl, val_dl, test_dl = split_dataset(cfg, ds, collate_fn=None)
        print(f"Found {len(train_ds)} training images and {len(val_ds) if val_ds else 0} validation images.")
        print(f"Training iterations per epoch: {len(train_dl)}, Validation iterations per epoch {len(val_dl) if val_dl else 0}")
        print(f"Available classes: {classes}")
    else:
        train_ds, val_ds, test_ds, train_dl, val_dl, test_dl = split_ft_dataset(cfg, ds)
        print(f"Found {len(train_ds)} training images and {len(val_ds) if val_ds else 0} validation images.")
        print(f"Training iterations per epoch: {len(train_dl)}, Validation iterations per epoch {len(val_dl) if val_dl else 0}")
        print(f"Available classes: {classes}")
    
    return train_ds, val_ds, test_ds, train_dl, val_dl, test_dl, classes, class_to_idx

def get_pretrained_model(cfg: Dict, classes: List[str]) -> torch.nn.Module:
    if cfg.pretrain == "imagenet":
        if cfg.arch == "vit_base":
            model = timm.create_model('vit_base_patch8_224.augreg2_in21k_ft_in1k', pretrained=True)
            model.head = torch.nn.Identity()
            print(f"Timm pretrained model loaded: {cfg.arch}")
        elif cfg.arch == "vit_small":
            model = timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
            model.head = torch.nn.Identity()
            print(f"Timm pretrained model loaded: {cfg.arch}")
        elif cfg.arch == "resnet":
            model = timm.create_model('resnet50.a1_in1k', pretrained=True)
            model.embed_dim = model.fc.in_features
            model.fc = torch.nn.Identity()
            print(f"Timm pretrained model loaded: {cfg.arch}")
        else:
            raise ValueError(f"Architecture {cfg.arch} not available pretrained on agm dataset.")
    elif cfg.pretrain == "agm":
        if cfg.arch in ["vit_base", "vit_small"]:
            model_ = vits.__dict__[cfg.arch](patch_size=cfg.patch_size, img_size=cfg.img_size)
            checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu")
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith("0."):
                    if k.startswith("0.head"):
                        continue
                    else:
                        new_state_dict[k[2:]] = v
                elif k.startswith("1."):
                    continue
            model_.load_state_dict(new_state_dict)
            print(f"Loaded checkpoint from {cfg.checkpoint_path}")
            
            if not cfg.only_cls_token:
                model = lambda x: model_(x, return_cls=False)[:,1:]
            else:
                model = model_
        elif cfg.arch == "resnet":
            model = resnet50(pretrained=False)
            model.embed_dim = model.fc.in_features
            model.fc = torch.nn.Identity()
            state_dict = torch.load(cfg.checkpoint_path)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint from {cfg.checkpoint_path}")

        else:
            raise ValueError(f"Architecture {cfg.arch} not available pretrained on agm dataset.")
    else:
        raise ValueError(f"Pretraining {cfg.pretrain} not supported; supported pretrain are 'agm' and 'imagenet'.")

    # attach new head
    head = vits.MLP(model.embed_dim, len(classes), cfg.mlp_num_layers, cfg.mlp_hidden_size)
    model = torch.nn.Sequential(collections.OrderedDict([('model', model), ('head', head)]))

    return model

# Function to train an SVM on the extracted features
def train_svm(features, labels):
    svm_classifier = SVC()
    svm_classifier.fit(features, labels)
    return svm_classifier

# Function to train a kNN classifier on the extracted features
def train_knn(features, labels, n_neighbors):
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(features, labels)
    return knn_classifier

def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, targets = data['image'], data['label']
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs)
            labels.append(targets)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels

def log_to_file(log_file, message):
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def run_classifier(feature_extractor, train_dataloader, val_dataloader, log_dir, run_id, device,
                     n_neighbors=3, svm=True, knn=True):

    log_file = os.path.join(log_dir, f'fine_tuning{run_id}.log')    

    print(f"Extracting features for {'svm' if svm else ''}{'knn' if knn else ''} classifier")
    train_features, train_labels = extract_features(feature_extractor, train_dataloader, device)
    val_features, val_labels = extract_features(feature_extractor, val_dataloader, device)

    # bring to cpu
    train_features = train_features.detach().cpu()
    train_labels = train_labels.detach().cpu()
    val_features = val_features.detach().cpu()
    val_labels = val_labels.detach().cpu()

    if svm:
        print("Training an SVM")
        # Train an SVM
        svm_classifier = train_svm(train_features, train_labels)

        # Evaluate on validation set
        svm_val_predictions = svm_classifier.predict(val_features)
        svm_val_accuracy = accuracy_score(val_labels, svm_val_predictions)
        log_to_file(log_file, f"SVM Validation Accuracy: {svm_val_accuracy}")

    if knn:
        print("Training a kNN classifier")
        # Train a kNN classifier
        knn_classifier = train_knn(train_features, train_labels, n_neighbors=n_neighbors)

        # Evaluate on validation set
        knn_val_predictions = knn_classifier.predict(val_features)
        knn_val_accuracy = accuracy_score(val_labels, knn_val_predictions)
        log_to_file(log_file, f"kNN Validation Accuracy: {knn_val_accuracy}")

