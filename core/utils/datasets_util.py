import os
from glob import glob
from typing import List, Tuple, Dict
from PIL import Image
from torch.utils.data import Dataset


class CustomImageFolder(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.samples = self._load_samples()
        self.class_to_idx = self._find_classes()
        self.classes = list(self.class_to_idx.keys())

    def _load_samples(self):
        samples = []
        classes = os.listdir(self.root)
        classes.sort()

        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(self.root, class_name)
            if not os.path.isdir(class_dir):
                continue

            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                samples.append((image_path, class_idx))

        return samples
    
    def _find_classes(self):
        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]

        with open(image_path, "rb") as f:
            image = Image.open(f).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'image': image,'label': target}
        
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
            return {'image': self.transform(img),'label': class_idx}
        else:
            return {'image': img,'label': class_idx}
        
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
            return {'image': self.transform(img),'label': class_idx}
        else:
            return {'image': img,'label': target}