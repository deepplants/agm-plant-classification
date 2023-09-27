import torch
from torch import nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
# import hydra
import omegaconf
from omegaconf import DictConfig
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import os
import sys
import argparse
import warnings
from torchvision import transforms
from PIL import Image
from core.model import vision_transformer as vits
from core.utils.fine_tune_utils import ImageFolderFT, ImageFolderFT_OneClass, find_classes
from core.utils.ft import main_ft
from torch.utils.data import random_split, DataLoader
from time import time
warnings.filterwarnings("ignore", category=UserWarning)


def main(cfg: DictConfig) -> None:
    t0 = time()
    if isinstance(cfg.fine_tune_on, str):
        if cfg.fine_tune_on == "all":
            _, _, classes, _ = find_classes(cfg.ds_dir)
            for specie in classes:
                t1 = time()
                print(f"--- {specie}")
                run(cfg, specie)
                t2 = time()
                print(f"--- {specie} took {t2-t1} seconds")
                print("Total time elapsed: ", t2-t0)
        else:
            run(cfg, cfg.fine_tune_on)
    elif isinstance(cfg.fine_tune_on, omegaconf.listconfig.ListConfig):
        for specie in cfg.fine_tune_on:
            run(cfg, specie)

def get_optimization_parameters(model, blocks_to_optimize):
    params_to_optimize = []
    for name, param in model.named_parameters():
        if isinstance(blocks_to_optimize, str):
            if blocks_to_optimize == "all":
                param.requires_grad = True
                params_to_optimize.append(param)
                # print("\t Head: ",name)
            elif name.startswith(blocks_to_optimize):
                param.requires_grad = True
                params_to_optimize.append(param)
                print("\t Head: ",name)
            else:
                param.requires_grad = False
        elif isinstance(blocks_to_optimize, omegaconf.listconfig.ListConfig):
            if any([name.startswith(block) for block in blocks_to_optimize]):
                param.requires_grad = True
                params_to_optimize.append(param)
            else:
                param.requires_grad = False

    return params_to_optimize

        
def run(cfg: DictConfig, specie) -> None:
    
        print(cfg)
        print(f"Fine tuning on {specie}")
    
        # prepare logging and checkpointing
        os.makedirs(cfg.result_dir, exist_ok=True)
        experiment_result_dir = os.path.join(cfg.result_dir, cfg.exp_id)
        os.makedirs(experiment_result_dir, exist_ok=True)
    
        log_file = os.path.join(experiment_result_dir, f"results_{specie}.csv")
        conf_log_file = os.path.join(experiment_result_dir, f"conf_{specie}.yaml")
        train_ds_log_file = os.path.join(experiment_result_dir, f"train_ds_{specie}.txt")
        val_ds_log_file = os.path.join(experiment_result_dir, f"val_ds_{specie}.txt")
        test_ds_log_file = os.path.join(experiment_result_dir, f"test_ds_{specie}.txt")

        transform = transforms.Compose([
            transforms.Resize(cfg.img_size, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(cfg.ds_mean, cfg.ds_std)
        ])

        # load dataset
        if cfg.fine_tune_on == "whole":          
            ds = ImageFolderFT(cfg.ds_dir, transform=transform, species_to_merge=cfg.species_to_merge) 
            classes = ds.classes
            paths = ds.paths
        elif cfg.fine_tune_on == "species":
            from torchvision.datasets import ImageFolder
            ds = ImageFolder(cfg.ds_dir, transform=transform)
            classes = ds.classes
            paths = np.array([x[0] for x in ds.imgs])
        elif cfg.fine_tune_on == "plant_doc":
            from fine_tune_utils import ImageFolderPlantDoc
            ds = ImageFolderPlantDoc(cfg.ds_dir, transform=transform)
            classes = ds.classes
            paths = ds.paths
        elif cfg.fine_tune_on == "plant_doc_multilabel":
            from fine_tune_utils import ImageFolderPlantDocMultiLabel
            ds = ImageFolderPlantDocMultiLabel(cfg.ds_dir, transform=transform)
            raise NotImplementedError("Multilabel fine tuning not supported yet")
        elif cfg.fine_tune_on == "cassava":
            from torchvision.datasets import ImageFolder
            ds = ImageFolder(cfg.ds_dir, transform=transform)
            classes = ds.classes
            paths = np.array([x[0] for x in ds.imgs])
        else:
            ds = ImageFolderFT_OneClass(cfg.ds_dir, specie, transform=transform, species_to_merge=cfg.species_to_merge)
            classes = ds.classes
            paths = np.array(ds.paths)
        
        #repeat dataset
        if cfg.repeat_ds:
            ds = torch.utils.data.ConcatDataset([ds]*cfg.repeat_ds)
            paths = np.concatenate([paths]*cfg.repeat_ds)

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

        # log dataset splits
        with open(train_ds_log_file, "w") as f:
            f.write("\n".join(paths[train_ds.indices]))
        with open(val_ds_log_file, "w") as f:
            f.write("\n".join(paths[val_ds.indices]))
        if cfg.test:
            with open(test_ds_log_file, "w") as f:
                f.write("\n".join(paths[test_ds.indices]))

        # log configuration
        with open(conf_log_file, "w") as f:
            f.write(omegaconf.OmegaConf.to_yaml(cfg))

        if not cfg.simple_eval:
            with open(log_file, "w") as f:
                f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

        
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

        

        if cfg.checkpoint_path == "base":
            import timm
            model = timm.create_model('vit_base_patch8_224.augreg2_in21k_ft_in1k', pretrained=True)
            print(f"Timm pretrained model loaded from {cfg.checkpoint_path}")
        elif cfg.checkpoint_path == "small":
            import timm
            model = timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
            print(f"Timm pretrained model loaded from {cfg.checkpoint_path}")

            # get model specific transforms (normalization, resize)
            # data_config = timm.data.resolve_model_data_config(model)
            # transforms_ = timm.data.create_transform(**data_config, is_training=True)
            # print(data_config)
            # print(model)
            # sys.exit()
        else:
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

        # model.eval()

        if cfg.simple_eval:
            # move model to device
            model = model.to(cfg.device)
            
            main_ft(
                model, 
                train_dl,
                val_dl,
                experiment_result_dir,
                specie,
                cfg.device,
                num_classes = len(classes),
                batch_size = cfg.batch_size,
                num_epochs = cfg.epochs,
                learning_rate = cfg.learning_rate,
                n_neighbors = cfg.n_neighbors,
                linear_layer = cfg.linear,
                svm = cfg.svm,
                knn = cfg.knn,
            )
            return

        # attach new head
        model.head = vits.MLP(model.embed_dim, len(classes), cfg.mlp_num_layers, cfg.mlp_hidden_size)  

        # move model to device
        model = model.to(cfg.device)
        print(f"Training model on {cfg.device}")
        model.head = model.head.to(cfg.device)
        print(f"Training head on {cfg.device}")
        if cfg.blocks_to_optimize:
            print(f"Optimizing blocks: {cfg.blocks_to_optimize}")
            params_to_optimize = get_optimization_parameters(model, cfg.blocks_to_optimize)
        else:
            params_to_optimize = list(model.parameters()) + list(model.head.parameters())
        # sys.exit()

        # create optimizer
        optimizer = optim.Adam(params_to_optimize, lr=cfg.learning_rate)

        # create loss function
        loss_fn = nn.CrossEntropyLoss()

        # train
        best_val_acc = 0
        for epoch in range(cfg.epochs):
            print(f"Epoch {epoch+1}/{cfg.epochs}")
            # train
            model.train()
            train_loss = 0
            train_acc = 0
            for x, y in tqdm(train_dl):
                x = x.to(cfg.device)
                y = y.to(cfg.device)
                optimizer.zero_grad()
                if cfg.pretrain == "imagenet":
                    y_hat = model(x) # for imagenet pt SUP model
                else:
                    feats = model(x)
                    y_hat = model.head(feats)
                loss = loss_fn(y_hat, y.long())
                loss.backward()
                optimizer.step()
                _, preds = torch.max(y_hat, 1)
                train_loss += loss.item()
                train_acc += torch.sum(preds == y.data)
            train_loss /= len(train_ds)
            train_acc = train_acc.item() / len(train_ds)

            # validate
            model.eval()
            val_loss = 0
            val_acc = 0
            y_true = []
            y_pred = []
            for x, y in val_dl:
                x = x.to(cfg.device)
                y = y.to(cfg.device)
                with torch.no_grad():
                    if cfg.pretrain == "imagenet":
                        y_hat = model(x) # for imagenet pt SUP model
                    else:
                        feats = model(x)
                        y_hat = model.head(feats)
                    loss = loss_fn(y_hat, y)
                    _, preds = torch.max(y_hat, 1)
                    val_loss += loss.item()
                    val_acc += torch.sum(preds == y.data)
                    y_true.extend(y.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())
            val_loss /= len(val_ds)
            val_acc = val_acc.item() / len(val_ds)
            # Compute confusion matrix
            if epoch % cfg.cm_every_n == 0:
                cm = confusion_matrix(y_true, y_pred)
                # normalize confusion matrix row wise
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                # Plot confusion matrix
                plt.figure(figsize=(10, 8))
                plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
                plt.colorbar()
                n_classes = range(len(classes))  # Assuming you have a list of class names
                tick_marks = np.arange(len(classes))
                plt.xticks(tick_marks, classes)
                plt.yticks(tick_marks, classes)
                plt.xlabel("Predicted labels")
                plt.ylabel("True labels")
                plt.title("Confusion Matrix")
                # for i in range(len(classes)):
                #     for j in range(len(classes)):
                for i in range(len(cm)):
                    for j in range(len(cm[0])):
                        plt.text(j, i, str(cm[i, j]), horizontalalignment="center", verticalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
                plt.tight_layout()
                plt.savefig(os.path.join(experiment_result_dir, f"confusion_matrix_epoch_{epoch}.png"))
                plt.close()
            # save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.head.state_dict(), os.path.join(experiment_result_dir, f"best_model_s{specie}.pth"))
                print(f"New best model saved at epoch {epoch} with validation accuracy {val_acc} and validation loss {val_loss}")
            
            # log
            with open(log_file, "a") as f:
                f.write(f"{epoch},{train_loss},{train_acc},{val_loss},{val_acc}\n")
            print(f"Epoch {epoch}: train loss {train_loss}, train accuracy {train_acc}, validation loss {val_loss}, validation accuracy {val_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()
    CONFIGURATION_FILENAME = args.cfg
    config = omegaconf.OmegaConf.load(CONFIGURATION_FILENAME)
    main(config)