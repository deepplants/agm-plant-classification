import torch
from torch import nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import omegaconf
from omegaconf import DictConfig
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import os
import sys
import argparse
import warnings
from core.model import vision_transformer as vits
from core.utils.fine_tune_utils import find_classes, get_finetune_sets, get_pretrained_model, run_knn
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

        train_ds, val_ds, test_ds, train_dl, val_dl, test_dl, classes, class_to_idx = get_finetune_sets(cfg)
        
        # log configuration
        with open(conf_log_file, "w") as f:
            f.write(omegaconf.OmegaConf.to_yaml(cfg))

        if not cfg.simple_eval:
            with open(log_file, "w") as f:
                f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

        if cfg.knn:
            # move model to device
            model = model.to(cfg.device)
            
            run_knn(
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
        
        model = get_pretrained_model(cfg, classes)

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