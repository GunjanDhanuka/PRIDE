import sys

sys.path.append("..")
from torch.utils.data import DataLoader, Dataset
import os
from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import NamedTuple
import pickle
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
import warnings

warnings.simplefilter("ignore")

from Models.disentangled_attention import LSTMDisentangledAttention
from Models.mlp import MLP, MMLP
from Models.mil import MILModel, ModifiedMILModel
from Utils.utils import seed_everything, process_feat, seg2frame, save_predictions
from Losses.losses import MIL
from DataLoaders.dataloaders import Normal_Loader, Anomaly_Loader

seed_everything(40)

class RGBModel(NamedTuple):
    model_name: str
    data_path: str
    output_size: int
    need_process: bool


def train(epoch, model, normal_train_loader, anomaly_train_loader, optimizer):
    print("\nEpoch: %d" % epoch)
    model.train()
    train_loss = 0
    loader = zip(normal_train_loader, anomaly_train_loader)
    pbar = tqdm(enumerate(loader, 1))
    for batch_idx, (normal_inputs, anomaly_inputs) in pbar:
        loss = model(anomaly_inputs, normal_inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # print(f"Processed [{batch_idx}]/[{min(len(normal_train_loader), len(anomaly_train_loader))}] MIL Loss : {loss.item():.4f}")
        pbar.set_description(
            f"Processed [{batch_idx}]/[{min(len(normal_train_loader), len(anomaly_train_loader))}] MIL Loss : {loss.item():.4f}"
        )

    pbar.clear()
    pbar.write("loss = {}".format(train_loss / len(normal_train_loader)))

    return train_loss / len(normal_train_loader)


def test(epoch, model, anomaly_test_loader, normal_test_loader):
    model.eval()
    auc = 0
    all_gt_list = []
    all_score_list = []
    with torch.no_grad():
        for i, (data) in enumerate(anomaly_test_loader):
            inputs, gts, frames, name = data
            score = model(inputs)
            score = score.cpu().detach().numpy()
            score_list, gt_list = seg2frame(frames.item(), score, gts)
            all_gt_list.extend(gt_list)
            all_score_list.extend(score_list)

        for i, (data2) in enumerate(normal_test_loader):
            inputs2, gts2, frames2, name2 = data2
            score2 = model(inputs2)
            score2 = score2.cpu().detach().numpy()
            score_list2, gt_list2 = seg2frame(frames2.item(), score2, [])
            all_gt_list.extend(gt_list2.tolist())
            all_score_list.extend(score_list2.tolist())

    fpr, tpr, thresholds = metrics.roc_curve(all_gt_list, all_score_list, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    base_path = get_original_cwd()
    os.chdir(base_path)
    print(OmegaConf.to_yaml(cfg))

    i3d = RGBModel(
        "i3d",
        cfg.dataset.i3d_path,
        1024,
        False,
    )

    swin = RGBModel(
        "swin",
        cfg.dataset.swin_path,
        1024,
        False,
    )

    s3d = RGBModel("s3d", cfg.dataset.s3d_path, 1024, True)

    rgb_models_all = [[i3d, swin, s3d]]
    # rgb_models_all = [[i3d, swin, s3d]]
    for rgb_models in rgb_models_all:
        mlp_size = 0
        for model in rgb_models:
            mlp_size += model.output_size
        mlp_hidden_dim = 1024
        mlp_output_dim = 512

        if mlp_size <= 1024:
            mlp_hidden_dim = 512

        data_path = cfg.dataset.path
        normal_train_dataset = Normal_Loader(rgb_models, is_train=1, path=data_path)
        normal_test_dataset = Normal_Loader(rgb_models, is_train=0, path=data_path)

        anomaly_train_dataset = Anomaly_Loader(rgb_models, is_train=1, path=data_path)
        anomaly_test_dataset = Anomaly_Loader(rgb_models, is_train=0, path=data_path)

        TRAIN_BATCH_SIZE = cfg.stage1.train_batch_size
        VALID_BATCH_SIZE = cfg.stage1.valid_batch_size
        normal_train_loader = DataLoader(
            normal_train_dataset,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            drop_last=True,
        )
        normal_pred_loader = DataLoader(
            normal_train_dataset,
            batch_size=VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=4,
        )
        normal_test_loader = DataLoader(
            normal_test_dataset,
            batch_size=VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=4,
        )

        anomaly_train_loader = DataLoader(
            anomaly_train_dataset,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            drop_last=True,
        )
        anomaly_pred_loader = DataLoader(
            anomaly_train_dataset,
            batch_size=VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=4,
        )
        anomaly_test_loader = DataLoader(
            anomaly_test_dataset,
            batch_size=VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=4,
        )

        device = cfg.stage1.device
        mlp_rgb = MLP(
            input_dim=mlp_size, hidden_dim=mlp_hidden_dim, output_dim=mlp_output_dim
        )
        mlp_flow = MLP(input_dim=1024, hidden_dim=512, output_dim=512)

        mil_model = ModifiedMILModel(input_dim=1024, hidden_dim1=512)

        t_model = LSTMDisentangledAttention()

        criterion = MIL
        model = MILModel(mlp_rgb, mlp_flow, mil_model, t_model, criterion, device).to(
            device
        )

        params = [
            {"params": model.mlp_rgb.parameters(), "lr": cfg.stage1.lr_mlp_rgb},
            {"params": model.mlp_flow.parameters(), "lr": cfg.stage1.lr_mlp_flow},
            {"params": model.mil_model.parameters()},
            {"params": model.t_model.parameters(), "lr": cfg.stage1.lr_t_model},
        ]
        optimizer = torch.optim.Adagrad(
            params, lr=cfg.stage1.lr, weight_decay=cfg.stage1.weight_decay
        )

        best_auc = 0.5
        auc_scores = []
        n_epochs = cfg.stage1.epochs
        pbar_main = tqdm(range(0, n_epochs))
        for epoch in pbar_main:
            epoch_train_loss = train(
                epoch, model, normal_train_loader, anomaly_train_loader, optimizer
            )
            pred_dict = save_predictions(
                epoch, model, anomaly_pred_loader, normal_pred_loader
            )
            epoch_auc = test(epoch, model, anomaly_test_loader, normal_test_loader)
            auc_scores.append(epoch_auc)
            if epoch_auc > best_auc:
                best_auc = epoch_auc
                best_pred_dict = pred_dict
            pbar_main.write(f"The AUC SCORE is {epoch_auc}")
            pbar_main.write(f"The BEST AUC SCORE for MIL Model is {best_auc}")

        save_file_name = (
            "_".join([model.model_name for model in rgb_models]) + ".pickle"
        )
        # with open(save_file_name, "wb") as f:
        #     pickle.dump(best_pred_dict, f)


if __name__ == "__main__":
    # python main_backbones.py --multirun t_model=gru,rnn,lstm,att,gru_att,dis_att,gru_dis_att,rnn_dis_att,tconv,rtfm
    main()
