import os
from typing import Dict, List, Tuple
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import numpy as np
from torchvision import transforms
from .train_util import load_from_url_or_disk, split_dataset
from datasets import ClassLabel

# Make function to find classes in target directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.
    
    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    species = []
    for i, class_ in enumerate(classes):
        if i == 0:
            species = sorted(entry.name for entry in os.scandir(os.path.join(directory, class_)) if entry.is_dir())
        else:
            hs = sorted(entry.name for entry in os.scandir(os.path.join(directory, class_)) if entry.is_dir())
            if hs != species:
                raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
            else:
                continue

    # species = [item for sublist in species for item in sublist]
    # species = sorted(set(species))
    
    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    # 3. Crearte a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    species_to_idx = {specie_name: i for i, specie_name in enumerate(species)}
    return classes, class_to_idx, species, species_to_idx

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

    # In order to reduce fine tuning time we use only test_spli M images
    if cfg.fine_tune_on == "species":
        train_ds = test_ds
        train_dl = test_dl

    print(f"Train iterations per epoch: {len(train_dl)}")
    print(f"Validation iterations per epoch: {len(val_dl)}")

def get_finetune_sets(cfg: Dict):

        transform = transforms.Compose([
            transforms.Resize(cfg.img_size, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(cfg.ds_mean, cfg.ds_std)
        ])

        # load dataset
        if cfg.fine_tune_on == "agmhs":          
            ds = load_from_url_or_disk(cfg.ds_remote_repo, cfg.ds_local_repo, cfg.ds_save_to_local)
            class_names = sorted(ds['train'].unique("label"))
            class_to_idx = ClassLabel(names=class_names)._str2int
            classes = list(class_to_idx.keys())
        elif cfg.fine_tune_on == "agm":
            ds = load_from_url_or_disk(cfg.ds_remote_repo, cfg.ds_local_repo, cfg.ds_save_to_local)
            class_names = sorted(ds['train'].unique("label"))
            class_to_idx = ClassLabel(names=class_names)._str2int
            classes = list(class_to_idx.keys())
        elif cfg.fine_tune_on == "plant_doc":
            ds = ImageFolderPlantDoc(cfg.ds_dir, transform=transform)
            classes = ds.classes
        elif cfg.fine_tune_on == "plant_doc_multilabel":
            ds = ImageFolderPlantDocMultiLabel(cfg.ds_dir, transform=transform)
            raise NotImplementedError("Multilabel fine tuning not supported yet")
        elif cfg.fine_tune_on == "cassava":
            from torchvision.datasets import ImageFolder
            ds = ImageFolder(cfg.ds_dir, transform=transform)
            classes = ds.classes
        else:
            raise ValueError(f"Fine tune dataset {cfg.fine_tune_on} not supported.")
        
        if cfg.fine_tune_on in ["agm", "agmhs"]:
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
    if cfg.checkpoint_path == "base":
        import timm
        model = timm.create_model('vit_base_patch8_224.augreg2_in21k_ft_in1k', pretrained=True)
        print(f"Timm pretrained model loaded from {cfg.checkpoint_path}")
    elif cfg.checkpoint_path == "small":
        import timm
        model = timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
        print(f"Timm pretrained model loaded from {cfg.checkpoint_path}")
    else:
        from ..model import vision_transformer as vits
        model_ = vits.__dict__[cfg.arch](patch_size=cfg.patch_size, img_size=cfg.img_size)

        # load pretrained weights
        if cfg.pretrain == "agm":
            if cfg.checkpoint_path:
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
        elif cfg.pretrain == "imagenet":
            state_dict = torch.load(cfg.checkpoint_path)
            model_.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint from {cfg.checkpoint_path}")

        if not cfg.only_cls_token:
            model = lambda x: model_(x, return_cls=False)[:,1:]
        else:
            model = model_

    # attach new head
    model.head = vits.MLP(model.embed_dim, len(classes), cfg.mlp_num_layers, cfg.mlp_hidden_size)

    return model

def run_knn():
    pass


# 1. Subclass torch.utils.data.Dataset
class ImageFolderFT(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None, species_to_merge=None) -> None:
        
        # 3. Create class attributes
        # Get all image paths
        self.paths = self.get_image_paths(targ_dir) # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx, self.species, self.species_to_idx = find_classes(targ_dir)

        self.species_to_merge = species_to_merge

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path) 
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name = self.paths[index].split(os.sep)[-3] # expects path in data_folder/class_name/specie/image.jpeg
        class_idx = self.class_to_idx[class_name]
        specie_name  = self.paths[index].split(os.sep)[-2]
        specie_idx = self.species_to_idx[specie_name]
        if self.species_to_merge is not None:
            for key, value in self.species_to_merge.items():
                if specie_name == value:
                    specie_idx = self.species_to_idx[key]
                    break

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx, specie_idx # return data, label (X, y)
        else:
            return img, class_idx, specie_idx # return data, label (X, y)
        
    def get_image_paths(self, directory: str) -> List[str]:
        image_paths = []
        supported_extensions = ['*.jpg'] # , '*.jpeg', '*.png', '*.gif']  # Add more extensions if needed

        for root, _, _ in os.walk(directory):
            for extension in supported_extensions:
                image_paths.extend(glob(os.path.join(root, extension)))

        return image_paths


class ImageFolderFT_OneClass(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, specie: str, transform=None, species_to_merge=None) -> None:
        
        self.species_to_merge = []
        if species_to_merge is not None:
            for key, value in species_to_merge.items():
                if specie == key:
                    for val in value:
                        self.species_to_merge.append(val)
                # if specie == value :
                #     self.species_to_merge.append(key)
                # elif specie == key:
                #     self.species_to_merge.append(value)
                else:
                    continue
                
                    
                    
        self.classes = [f for f in os.listdir(targ_dir) if os.path.isdir(os.path.join(targ_dir, f))]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.paths = self.get_image_paths(targ_dir, specie) # note: you'd have to update this if you've got .png's or .jpeg's
        if self.species_to_merge:
            for sp in self.species_to_merge:
                self.paths.extend(self.get_image_paths(targ_dir, sp))
        # Setup transforms
        self.transform = transform       
        self.species_to_merge = species_to_merge


    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path) 
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name = self.paths[index].split(os.sep)[-3] # expects path in data_folder/class_name/specie/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)
        
    def get_image_paths(self, directory: str, specie: str) -> List[str]:
        image_paths = []
        
        for class_ in self.classes:
            image_paths.extend(glob(os.path.join(directory, class_, specie, '*.jpg')))

        return image_paths
    
class ImageFolderSpecies(Dataset):
        
        # 2. Initialize with a targ_dir and transform (optional) parameter
        def __init__(self, targ_dir: str, transform=None) -> None:
            
            # 3. Create class attributes
            # Get all image paths
            self.paths = self.get_image_paths(targ_dir) # note: you'd have to update this if you've got .png's or .jpeg's
            # Setup transforms
            self.transform = transform
            # Create classes and class_to_idx attributes
            self.classes, self.class_to_idx, self.species, self.species_to_idx = find_classes(targ_dir)
    
        # 4. Make function to load images
        def load_image(self, index: int) -> Image.Image:
            "Opens an image via a path and returns it."
            image_path = self.paths[index]
            return Image.open(image_path) 
        
        # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
        def __len__(self) -> int:
            "Returns the total number of samples."
            return len(self.paths)
        
        # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
        def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
            "Returns one sample of data, data and label (X, y)."
            img = self.load_image(index)
            class_name = self.paths[index].split(os.sep)[-3] # expects path in data_folder/class_name/specie/image.jpeg
            class_idx = self.class_to_idx[class_name]
            specie_name  = self.paths[index].split(os.sep)[-2]
            specie_idx = self.species_to_idx[specie_name]
    
            # Transform if necessary
            if self.transform:
                return self.transform(img), class_idx, specie_idx # return data, label (X, y)
            else:
                return img, class_idx, specie_idx # return data, label (X, y)
            
        def get_image_paths(self, directory: str) -> List[str]:
            image_paths = []
            supported_extensions = ['*.jpg'] # , '*.jpeg', '*.png', '*.gif']  # Add more extensions if needed
    
            for root, _, _ in os.walk(directory):
                for extension in supported_extensions:
                    image_paths.extend(glob(os.path.join(root, extension)))
            
            return image_paths
        
class ImageFolderPlantDoc(Dataset):
    def __init__(self, targ_dir: str, transform=None, split="train"):
        self.paths = glob(os.path.join(targ_dir, split, "*.jpg"))
        clss = os.path.join(targ_dir, "train", "_classes.txt")
        with open(clss, "r") as f:
            classes = f.read().splitlines()
        self.classes = classes[0].split(", ")[1:]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.classes)}
        labels = [cl.split(", ")[1:] for cl in classes[1:]]
        self.labels = []
        for i, lab in enumerate(labels):
            try:
                self.labels.append([self.class_to_idx[self.classes[lab.index("1")]]])
            except ValueError:
            # remove images that are not classifies
                self.paths.pop(i)
        self.transform = transform

    def load_image(self, index: int):
        image_path = self.paths[index]
        return Image.open(image_path).convert("RGB")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        img = self.load_image(index)
        class_idx = self.labels[index][0]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx
        
class ImageFolderPlantDocMultiLabel(Dataset):
    def __init__(self, targ_dir: str, transform=None, split="train"):
        # plant_doc_multiclass_to_multilabel_dict
        self.ml_dict = {
            'apple scab leaf': ('apple leaf', 'scab'),
            'apple leaf': ('apple leaf', 'none'),
            'apple rust leaf': ('apple leaf', 'rust'),
            'bell_pepper leaf': ('bell_pepper leaf', 'none'),
            'bell_pepper leaf spot': ('bell_pepper leaf', 'spot'),
            'blueberry leaf': ('blueberry leaf', 'none'),
            'cherry leaf': ('cherry leaf', 'none'),
            'corn gray leaf spot': ('corn leaf', 'gray spot'),
            'corn leaf blight': ('corn leaf', 'blight'),
            'corn rust leaf': ('corn leaf', 'rust'),
            'peach leaf': ('peach leaf', 'none'),
            'potato leaf': ('potato leaf', 'none'),
            'potato leaf early blight': ('potato leaf', 'early blight'),
            'potato leaf late blight':   ('potato leaf', 'late blight'),
            'raspberry leaf': ('raspberry leaf', 'none'),
            'soyabean leaf': ('soyabean leaf', 'none'),
            'soybean leaf': ('soyabean leaf', 'none'),
            'squash powdery mildew leaf': ('squash leaf', 'powdery mildew'),
            'strawberry leaf': ('strawberry leaf', 'none'),
            'tomato early blight leaf': ('tomato leaf', 'early blight'),
            'tomato septoria leaf spot': ('tomato leaf', 'septoria spot'),
            'tomato leaf': ('tomato leaf', 'none'),
            'tomato leaf bacterial spot': ('tomato leaf', 'bacterial spot'),
            'tomato leaf late blight': ('tomato leaf', 'late blight'),
            'tomato leaf mosaic virus': ('tomato leaf', 'mosaic virus'),
            'tomato leaf yellow virus': ('tomato leaf', 'yellow virus'),
            'tomato mold leaf': ('tomato leaf', 'mold'),
            'tomato two spotted spider mites leaf': ('tomato leaf', 'two spotted spider mites'),
            'grape leaf': ('grape leaf', 'none'),
            'grape leaf black rot': ('grape leaf', 'black rot')
        }

        self.species = list(set([x[0] for x in self.ml_dict.values()]))
        self.stresses = list(set([x[1] for x in self.ml_dict.values()]))
        self.species_to_idx = {s: i for i, s in enumerate(self.species)}
        self.stresses_to_idx = {s: i for i, s in enumerate(self.stresses)}
        self.idx_to_species = {i: s for i, s in enumerate(self.species)}
        self.idx_to_stresses = {i: s for i, s in enumerate(self.stresses)}

        self.paths = glob(os.path.join(targ_dir, split, "*.jpg"))

        clss = os.path.join(targ_dir, split, "_classes.txt")
        with open(clss, "r") as f:
            classes = f.read().splitlines()
        self.classes = classes[0].split(", ")[1:]
        self.class_to_idx = {cls_name.lower(): i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name.lower() for i, cls_name in enumerate(self.classes)}
        self.paths = [os.path.join(cl.split(", ")[0]) for cl in classes[1:]]
        labels = [cl.split(", ")[1:] for cl in classes[1:]]
        self.labels = []
        for i, lab in enumerate(labels):
            try:
                self.labels.append([self.class_to_idx[self.classes[lab.index("1")].lower()]])
            except ValueError:
            # remove images that are not classifies
                self.paths.pop(i)
        self.transform = transform

    def load_image(self, index: int):
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        img = self.load_image(index)
        class_idx = self.labels[index]
        class_ = self.idx_to_class[class_idx[0]]
        species, stress = self.ml_dict[class_]
        target = (self.species_to_idx[species], self.stresses_to_idx[stress])

        if self.transform:
            return self.transform(img), target
        else:
            return img, target