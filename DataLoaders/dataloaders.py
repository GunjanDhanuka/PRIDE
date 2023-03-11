import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from Utils.utils import process_feat, preprocess_score
import pickle
import torch


class Normal_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """

    def __init__(
        self,
        rgb_models,
        is_train=1,
        path="/local/scratch/c_adabouei/video_analysis/video_anomaly_detection/UCF_and_Shanghai/UCF-Crime",
        stage2=False,
    ):
        super(Normal_Loader, self).__init__()
        self.is_train = is_train
        self.path = path
        self.rgb_models = rgb_models
        self.stage2 = stage2
        # self.rgb_dir = os.path.join(self.path,"all_rgbs")
        self.flow_dir = os.path.join(self.path, "all_flows")
        # self.vit_patch_dir  = "/local/scratch/c_adabouei/video_analysis/video_anomaly_detection/UCF_Crime_features_timesformer"
        if self.is_train == 1:
            data_list = "files/train_normal.txt"
            with open(data_list, "r") as f:
                self.data_list = f.readlines()
        else:
            data_list = "files/test_normalv2.txt"
            with open(data_list, "r") as f:
                self.data_list = f.readlines()
            # self.data_list = self.data_list[:-10]
        if self.stage2:
            with open("files/i3d_swin_s3d.pickle", "rb") as f:
                score_dict = pickle.load(f)
            self.pseudo_labels = score_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_train == 1:
            # rgb_npy = np.load(os.path.join(self.rgb_dir, self.data_list[idx][:-1]+'.npy'))
            rgb_npy = []
            for model in self.rgb_models:
                model_name = model.model_name
                datapath = model.data_path
                req_process = model.need_process
                features = np.load(
                    os.path.join(datapath, self.data_list[idx][:-1] + ".npy")
                )
                if req_process:
                    features = process_feat(features, 32)
                rgb_npy.append(features)
            rgb_npy = np.concatenate(rgb_npy, axis=-1)
            flow_npy = np.load(
                os.path.join(self.flow_dir, self.data_list[idx][:-1] + ".npy")
            )
            # patch_npy = np.load(os.path.join(self.vit_patch_dir, self.data_list[idx][:-1]+'.npy'))
            name = self.data_list[idx][:-1]
            if self.stage2:
                scores = torch.tensor([0] * 32, dtype=torch.float32)
                stack_npy = rgb_npy, flow_npy, scores
                return stack_npy
            stack_npy = rgb_npy, flow_npy, name
            return stack_npy
        else:
            name, frames, gts = (
                self.data_list[idx].split(" ")[0],
                int(self.data_list[idx].split(" ")[1]),
                int(self.data_list[idx].split(" ")[2][:-1]),
            )
            # rgb_npy = np.load(os.path.join(self.rgb_dir, name + '.npy'))
            rgb_npy = []
            for model in self.rgb_models:
                model_name = model.model_name
                datapath = model.data_path
                req_process = model.need_process
                features = np.load(os.path.join(datapath, name + ".npy"))
                if req_process:
                    features = process_feat(features, 32)
                rgb_npy.append(features)
            rgb_npy = np.concatenate(rgb_npy, axis=-1)
            flow_npy = np.load(os.path.join(self.flow_dir, name + ".npy"))
            # patch_npy = np.load(os.path.join(self.vit_patch_dir, name +'.npy'))
            stack_npy = rgb_npy, flow_npy
            return stack_npy, gts, frames, name


class Anomaly_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """

    def __init__(
        self,
        rgb_models,
        is_train=1,
        path="/local/scratch/c_adabouei/video_analysis/video_anomaly_detection/UCF_and_Shanghai/UCF-Crime",
        stage2=False,
    ):
        super(Anomaly_Loader, self).__init__()
        self.is_train = is_train
        self.path = path
        self.rgb_models = rgb_models
        self.stage2 = stage2
        # self.rgb_dir = os.path.join(self.path, "all_rgbs")
        self.flow_dir = os.path.join(self.path, "all_flows")
        # self.vit_patch_dir  = "/local/scratch/c_adabouei/video_analysis/video_anomaly_detection/UCF_Crime_features_timesformer"
        if self.is_train == 1:
            data_list = "files/train_anomaly.txt"
            with open(data_list, "r") as f:
                self.data_list = f.readlines()
        else:
            data_list = "files/test_anomalyv2.txt"
            with open(data_list, "r") as f:
                self.data_list = f.readlines()
        if self.stage2:
            with open("files/i3d_swin_s3d.pickle", "rb") as f:
                score_dict = pickle.load(f)
            self.pseudo_labels = score_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_train == 1:
            # rgb_npy = np.load(os.path.join(self.rgb_dir, self.data_list[idx][:-1]+'.npy'))
            rgb_npy = []
            for model in self.rgb_models:
                model_name = model.model_name
                datapath = model.data_path
                req_process = model.need_process
                features = np.load(
                    os.path.join(datapath, self.data_list[idx][:-1] + ".npy")
                )
                if req_process:
                    features = process_feat(features, 32)
                rgb_npy.append(features)
            rgb_npy = np.concatenate(rgb_npy, axis=-1)
            flow_npy = np.load(
                os.path.join(self.flow_dir, self.data_list[idx][:-1] + ".npy")
            )
            # patch_npy = np.load(os.path.join(self.vit_patch_dir, self.data_list[idx][:-1]+'.npy'))
            if self.stage2:
                video_name = self.data_list[idx][:-1].split(os.path.sep)[-1]
                scores = torch.tensor(
                    preprocess_score(self.pseudo_labels[video_name], average=False),
                    dtype=torch.float32,
                )
                stack_npy = rgb_npy, flow_npy, scores
                return stack_npy
            name = self.data_list[idx][:-1]
            stack_npy = rgb_npy, flow_npy, name
            return stack_npy
        else:
            name, frames, gts = (
                self.data_list[idx].split("|")[0],
                int(self.data_list[idx].split("|")[1]),
                self.data_list[idx].split("|")[2][1:-2].split(","),
            )
            gts = [int(i) for i in gts]
            # rgb_npy = np.load(os.path.join(self.rgb_dir, name + '.npy'))
            rgb_npy = []
            for model in self.rgb_models:
                model_name = model.model_name
                datapath = model.data_path
                req_process = model.need_process
                features = np.load(os.path.join(datapath, name + ".npy"))
                if req_process:
                    features = process_feat(features, 32)
                rgb_npy.append(features)
            rgb_npy = np.concatenate(rgb_npy, axis=-1)
            flow_npy = np.load(os.path.join(self.flow_dir, name + ".npy"))
            # patch_npy = np.load(os.path.join(self.vit_patch_dir, name +'.npy'))
            stack_npy = rgb_npy, flow_npy
            return stack_npy, gts, frames, name
