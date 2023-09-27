import torch
from torch import nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
# import hydra
import omegaconf
from omegaconf import DictConfig
from datetime import datetime
from sklearn.metrics import confusion_matrix
from core.utils.train_util import get_sets, get_model, unnormalize, train_or_validate
import os
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import sys
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)


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
    train_ds, val_ds, test_ds, train_dl, val_dl, test_dl, classes, class_to_idx, class_weights = get_sets(cfg)

    print(f"Found {len(train_ds)} training images and {len(val_ds) if val_ds else 0} validation images.")
    print(f"Training iterations per epoch: {len(train_dl)}, Validation iterations per epoch {len(val_dl) if val_dl else 0}")
    print(f"Available classes: {classes}")

    # log hyperparameters
    with open(conf_log_file, "w") as f:
        # f.write(cfg.pretty())
        f.write(omegaconf.OmegaConf.to_yaml(cfg))

    # log dataset splits
    train_samples = [str(x['indices']) for x in train_ds._indices.to_pylist()]
    if val_ds:
        val_samples = [str(x['indices']) for x in val_ds._indices.to_pylist()]
    else:
        val_samples = [None]
    if test_ds:
        test_samples = [str(x['indices']) for x in test_ds._indices.to_pylist()]
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

    # Main training loop
    for epoch in range(start_epoch, end_epoch):
        print(f"Epoch {epoch}")

        train_loss, train_acc, train_precision, train_recall, train_f1, all_labels, all_preds = train_or_validate(model, train_dl, criterion, optimizer, device, is_training=True)

        if val_dl:
            val_loss, val_acc, val_precision, val_recall, val_f1, all_labels, all_preds = train_or_validate(model, val_dl, criterion, optimizer, device, is_training=False)
            
            if epoch % cfg.cm_every_n == 0:
                # Calculate and plot confusion matrix
                y_true = np.array(all_labels)
                y_pred = np.array(all_preds)

                if len(y_pred.shape) == 1:
                    cm = confusion_matrix(y_true, y_pred)
                elif len(y_pred.shape) == 2:
                    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))

                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                plt.figure(figsize=(10, 8))
                plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
                plt.colorbar()
                n_classes = range(len(classes))
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()
    CONFIGURATION_FILENAME = args.cfg
    config = omegaconf.OmegaConf.load(CONFIGURATION_FILENAME)
    if config.train:
        main(config)
    else:
        test(config)