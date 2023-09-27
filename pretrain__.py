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
from core.utils.train_util import get_sets, get_model, unnormalize
import os
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import sys
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

# os.environ["HYDRA_FULL_ERROR"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# CONFIGURATION_PATH = os.environ["CONFIGURATION_PATH"]
CONFIGURATION_FILENAME = os.environ["CONFIGURATION_FILENAME"]
CONFIGURATION_PATH = "/workdir/baseline_species/configs"

def main(cfg: DictConfig) -> None:

    pp.pprint(cfg)

    # prepare logging and checkpointing
    os.makedirs(cfg.result_dir, exist_ok=True)
    experiment_result_dir = os.path.join(cfg.result_dir, cfg.exp_id)
    cktp_dir = os.path.join(experiment_result_dir, "ckpts")
    os.makedirs(cktp_dir, exist_ok=True)

    log_file = os.path.join(experiment_result_dir, f"results_{cfg.exp_id}.csv")
    conf_log_file = os.path.join(experiment_result_dir, f"conf_{cfg.exp_id}.yaml")
    train_ds_log_file = os.path.join(experiment_result_dir, f"train_ds_{cfg.exp_id}.txt")
    val_ds_log_file = os.path.join(experiment_result_dir, f"val_ds_{cfg.exp_id}.txt")
    test_ds_log_file = os.path.join(experiment_result_dir, f"test_ds_{cfg.exp_id}.txt")

    # load dataset
    train_ds, val_ds, test_ds, train_dl, val_dl, test_dl, classes, samples, class_to_idx, class_weights = get_sets(cfg)

    print(f"Found {len(train_ds)} training images and {len(val_ds) if val_ds else 0} validation images.")
    print(f"Training iterations per epoch: {len(train_dl)}, Validation iterations per epoch {len(val_dl) if val_dl else 0}")
    print(f"Available classes: {classes}")

    # log hyperparameters
    with open(conf_log_file, "w") as f:
        # f.write(cfg.pretty())
        f.write(omegaconf.OmegaConf.to_yaml(cfg))

    # log dataset
    samples = np.array(samples)
    train_samples = samples[train_ds.indices][:, 0]
    if val_ds:
        val_samples = samples[val_ds.indices][:, 0]
    else:
        val_samples = [None]
    if test_ds:
        test_samples = samples[test_ds.indices][:, 0]
    else:
        test_samples = [None]

    with open(train_ds_log_file, "w") as f:
        f.write("\n".join(train_samples))
    with open(val_ds_log_file, "w") as f:
        f.write("\n".join(val_samples))
    with open(test_ds_log_file, "w") as f:
        f.write("\n".join(test_samples))   

    # load model
    model = get_model(cfg, classes) 

    # optionally resume training from a checkpoint
    if not cfg.resume_training:   
        if val_dl: 
            with open(log_file, "w") as f:
                f.write("epoch,train_loss,train_acc,train_precision,train_recall,train_f1,val_loss,val_acc,val_precision,val_recall,val_f1\n")
        else:
            with open(log_file, "w") as f:
                f.write("epoch,train_loss,train_acc,train_precision,train_recall,train_f1\n")
        start_epoch = 0
        end_epoch = cfg.epochs
    else:
        if cfg.checkpoint_path is None:
            raise ValueError("checkpoint_path must be specified when resuming training")
        model.load_state_dict(torch.load(cfg.checkpoint_path))
        last_epoch = int(os.path.basename(cfg.checkpoint_path).split("_")[1].split(".")[0])
        start_epoch = last_epoch + 1
        end_epoch = cfg.epochs
        print(f"Resuming training from epoch {start_epoch} to epoch {end_epoch} from {cfg.checkpoint_path}.")

    # train
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model.to(device)

    if cfg.use_weighted_loss:
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    weight_decay = cfg.opt_weight_decay if hasattr(cfg, "weight_decay") else 0.0
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=weight_decay)

    for epoch in range(start_epoch, end_epoch):
        print(f"Epoch {epoch}")
        running_loss = 0.0
        running_acc = 0.0
        running_precision = 0.0
        running_recall = 0.0
        for i, data in tqdm(enumerate(train_dl)):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            if len(labels.shape) == 1:
                running_acc += torch.sum(preds == labels.data)
                # Compute precision and recall
                running_precision += torch.sum(preds == labels.data).item()
                running_recall += len(preds)
            elif len(labels.shape) == 2:
                running_acc += torch.sum(preds == torch.argmax(labels, dim=1).data)  # Convert one-hot labels to indices
                # Compute precision and recall
                running_precision += torch.sum(preds == torch.argmax(labels, dim=1).data).item()  # Convert one-hot labels to indices
                running_recall += len(preds)

        train_loss = running_loss / len(train_dl)
        train_acc = running_acc / len(train_ds)
        train_precision = running_precision / running_recall
        train_recall = running_recall / len(train_ds)
        train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall)

        print(f"Epoch: {epoch}, Loss: {train_loss:.3f}, Acc: {train_acc:.3f}, Precision: {train_precision:.3f}, Recall: {train_recall:.3f}, F1: {train_f1:.3f}")

        # validation
        if val_dl:
            model.eval()
            val_running_loss = 0.0
            val_running_acc = 0.0
            val_running_precision = 0.0
            val_running_recall = 0.0
            y_true = []
            y_pred = []
            for i, data in tqdm(enumerate(val_dl)):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                if len(labels.shape) == 1:
                    val_running_acc += torch.sum(preds == labels.data)
                    # Compute precision and recall
                    val_running_precision += torch.sum(preds == labels.data).item()
                    val_running_recall += len(preds)
                elif len(labels.shape) == 2:
                    val_running_acc += torch.sum(preds == torch.argmax(labels, dim=1).data)
                    # Compute precision and recall
                    val_running_precision += torch.sum(preds == torch.argmax(labels, dim=1).data).item()
                    val_running_recall += len(preds)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

            if epoch % cfg.cm_every_n == 0:
                y_pred = np.array(y_pred)
                y_true = np.array(y_true)
                y_true = np.argmax(y_true, axis=1)
                print(y_pred.shape, y_true.shape)
                if len(y_pred.shape) == 1:
                    cm = confusion_matrix(y_true, y_pred)
                    # normalize confusion matrix row wise
                    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                elif len(y_pred.shape) == 2:
                    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
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
                for i in range(len(classes)):
                    for j in range(len(classes)):
                        plt.text(j, i, "{:.2f}".format(cm[i, j]), horizontalalignment="center", verticalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
                plt.tight_layout()
                plt.savefig(os.path.join(experiment_result_dir, f"confusion_matrix_epoch_{epoch}.png"))
                plt.close()
            
            val_loss = val_running_loss / len(val_dl)
            val_acc = val_running_acc / len(val_ds)
            val_precision = val_running_precision / val_running_recall
            val_recall = val_running_recall / len(val_ds)
            val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall)

            print(f"Epoch: {epoch}, Validation Loss: {val_loss:.3f}, Validation Acc: {val_acc:.3f}, Validation Precision: {val_precision:.3f}, Validation Recall: {val_recall:.3f}, Validation F1: {val_f1:.3f}")

        # save model
        if epoch % cfg.save_every_n == 0:
            torch.save(model.state_dict(), os.path.join(cktp_dir, f"model_{epoch}.pth"))
        # log loss to file
        with open(log_file, "a") as f:
            if val_dl:
                f.write(f"{epoch},{train_loss:.3f},{train_acc:.3f},{train_precision:.3f},{train_recall:.3f},{train_f1:.3f},{val_loss:.3f},{val_acc:.3f},{val_precision:.3f},{val_recall:.3f},{val_f1}\n")
            else:
                f.write(f"{epoch},{train_loss:.3f},{train_acc:.3f},{train_precision:.3f},{train_recall:.3f},{train_f1:.3f}\n")

        if epoch < end_epoch - 1:
            model.train()

def test(cfg: DictConfig):
    print(cfg.pretty())
    pass

if __name__ == "__main__":
    config = omegaconf.OmegaConf.load(os.path.join(CONFIGURATION_PATH, CONFIGURATION_FILENAME + ".yaml"))
    if config.train:
        main(config)
    else:
        test(config)