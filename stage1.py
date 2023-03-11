import sys
sys.path.append('..')
from torch.utils.data import DataLoader, Dataset
import os
from sklearn import metrics
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import NamedTuple
import pickle
from tqdm import tqdm
from Models.disentangled_attention import LSTMDisentangledAttention
from Utils.utils import seed_everything, process_feat, seg2frame
from Losses.binloss import Binloss
from Losses.mil import MIL

import warnings
warnings.simplefilter("ignore")
# THE RGB MODELS AND THEIR FEATURE SIZES:
# 1. I3D - 1024
# 2. SWIN - 1024
# 3. S3D - 1024
# 4. TIMESFORMER LARGE - 768
# 5. TIMESFORMER SMALL - 768

class RGBModel(NamedTuple):
    model_name: str
    data_path: str
    output_size: int
    need_process: bool

i3d = RGBModel('i3d', '/local/scratch/c_adabouei/video_analysis/video_anomaly_detection/UCF_and_Shanghai/UCF-Crime/all_rgbs', 1024, False)

swin = RGBModel('swin', '/local/scratch/c_adabouei/video_analysis/video_anomaly_detection/UCF_Crime_video_swin_features', 1024, False)

s3d = RGBModel('s3d', '/local/scratch/c_adabouei/video_analysis/dataset/UCF_Crime/S3D', 1024, True)

timesformer_large = RGBModel('timesformer_large', '/local/scratch/c_adabouei/video_analysis/video_anomaly_detection/UCF_Crime_features_timesformer_large', 768, False)

timesformer_small = RGBModel('transformer_small', '/local/scratch/c_adabouei/video_analysis/video_anomaly_detection/UCF_Crime_features_timesformer', 768, False)

# rgb_models = [i3d, swin, s3d, timesformer_large, timesformer_small]
    
seed_everything(40)

class Normal_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """
    def __init__(self, rgb_models, is_train=1,  path="/local/scratch/c_adabouei/video_analysis/video_anomaly_detection/UCF_and_Shanghai/UCF-Crime"):
        super(Normal_Loader, self).__init__()
        self.is_train = is_train
        self.path = path
        self.rgb_models = rgb_models
        # self.rgb_dir = os.path.join(self.path,"all_rgbs")
        self.flow_dir = os.path.join(self.path, "all_flows")
        # self.vit_patch_dir  = "/local/scratch/c_adabouei/video_analysis/video_anomaly_detection/UCF_Crime_features_timesformer"
        if self.is_train == 1:
            data_list = '../files/train_normal.txt'
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = '../files/test_normalv2.txt'
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
            # self.data_list = self.data_list[:-10]
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
                features = np.load(os.path.join(datapath, self.data_list[idx][:-1]+'.npy'))
                if req_process:
                    features = process_feat(features, 32)
                rgb_npy.append(features)
            rgb_npy = np.concatenate(rgb_npy, axis=-1)
            flow_npy = np.load(os.path.join(self.flow_dir, self.data_list[idx][:-1]+'.npy'))
            # patch_npy = np.load(os.path.join(self.vit_patch_dir, self.data_list[idx][:-1]+'.npy'))
            name = self.data_list[idx][:-1]
            stack_npy = rgb_npy, flow_npy, name
            return stack_npy
        else:
            name, frames, gts = self.data_list[idx].split(' ')[0], int(self.data_list[idx].split(' ')[1]), int(self.data_list[idx].split(' ')[2][:-1])
            # rgb_npy = np.load(os.path.join(self.rgb_dir, name + '.npy'))
            rgb_npy = []
            for model in self.rgb_models:
                model_name = model.model_name
                datapath = model.data_path
                req_process = model.need_process
                features = np.load(os.path.join(datapath, name +'.npy'))
                if req_process:
                    features = process_feat(features, 32)
                rgb_npy.append(features)
            rgb_npy = np.concatenate(rgb_npy, axis=-1)
            flow_npy = np.load(os.path.join(self.flow_dir, name + '.npy'))
            # patch_npy = np.load(os.path.join(self.vit_patch_dir, name +'.npy'))
            stack_npy = rgb_npy, flow_npy
            return stack_npy, gts, frames, name

class Anomaly_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """
    def __init__(self, rgb_models, is_train=1,  path="/local/scratch/c_adabouei/video_analysis/video_anomaly_detection/UCF_and_Shanghai/UCF-Crime"):
        super(Anomaly_Loader, self).__init__()
        self.is_train = is_train
        self.path = path
        self.rgb_models = rgb_models
        # self.rgb_dir = os.path.join(self.path, "all_rgbs")
        self.flow_dir = os.path.join(self.path, "all_flows")
        # self.vit_patch_dir  = "/local/scratch/c_adabouei/video_analysis/video_anomaly_detection/UCF_Crime_features_timesformer"
        if self.is_train == 1:
            data_list = '../files/train_anomaly.txt'
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = '../files/test_anomalyv2.txt'
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()

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
                features = np.load(os.path.join(datapath, self.data_list[idx][:-1]+'.npy'))
                if req_process:
                    features = process_feat(features, 32)
                rgb_npy.append(features)
            rgb_npy = np.concatenate(rgb_npy, axis=-1)
            flow_npy = np.load(os.path.join(self.flow_dir, self.data_list[idx][:-1]+'.npy'))
            # patch_npy = np.load(os.path.join(self.vit_patch_dir, self.data_list[idx][:-1]+'.npy'))
            name = self.data_list[idx][:-1]
            stack_npy = rgb_npy, flow_npy, name
            return stack_npy
        else:
            name, frames, gts = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1]), self.data_list[idx].split('|')[2][1:-2].split(',')
            gts = [int(i) for i in gts]
            # rgb_npy = np.load(os.path.join(self.rgb_dir, name + '.npy'))
            rgb_npy = []
            for model in self.rgb_models:
                model_name = model.model_name
                datapath = model.data_path
                req_process = model.need_process
                features = np.load(os.path.join(datapath,name +'.npy'))
                if req_process:
                    features = process_feat(features, 32)
                rgb_npy.append(features)
            rgb_npy = np.concatenate(rgb_npy, axis=-1)
            flow_npy = np.load(os.path.join(self.flow_dir, name + '.npy'))
            # patch_npy = np.load(os.path.join(self.vit_patch_dir, name +'.npy'))
            stack_npy = rgb_npy, flow_npy
            return stack_npy , gts, frames, name


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return x

class ModifiedMILModel(nn.Module):
    def __init__(self, input_dim = 1024, hidden_dim1 = 512, hidden_dim2 = 32, output_dim =1 , drop_p = 0.3):

        super(ModifiedMILModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, output_dim)
        self.dropout = nn.Dropout(drop_p)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight = nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    
    def forward(self, x):

        x = self.dropout(F.relu(self.bn1(self.linear1(x))))
        x = self.dropout(F.relu(self.bn2(self.linear2(x))))
        x = torch.sigmoid(self.linear3(x))

        return x


class MILModel(nn.Module):
    def __init__(self, mlp_rgb, mlp_flow, mil_model, t_model, criterion, device):
        super(MILModel, self).__init__()
        self.mlp_rgb = mlp_rgb
        self.mlp_flow = mlp_flow
        self.mil_model = mil_model
        self.t_model = t_model
        self.criterion = criterion
        self.device = device

    def forward(self, anomaly_inputs, normal_inputs = None):
        if normal_inputs is not None:
            rgb_normal, flow_normal, _ = normal_inputs
            rgb_normal, flow_normal = rgb_normal.to(self.device), flow_normal.to(self.device)
            # rgb_patch_normal = torch.cat([rgb_normal, patch_normal], dim = -1) 
            rgb_normal = self.mlp_rgb(rgb_normal)
            flow_normal = self.mlp_flow(flow_normal)

            # normalise the features
            rgb_normal = F.normalize(rgb_normal, p=2, dim=1)
            flow_normal = F.normalize(flow_normal, p=2, dim=1)
            normal_inputs = torch.cat([rgb_normal, flow_normal], dim = 2)
            normal_inputs = self.t_model(normal_inputs)

            rgb_anomaly, flow_anomaly, _ = anomaly_inputs
            rgb_anomaly, flow_anomaly = rgb_anomaly.to(self.device), flow_anomaly.to(self.device)
            # rgb_anomaly = torch.cat([rgb_anomaly, patch_anomaly], dim = -1) 
            rgb_anomaly = self.mlp_rgb(rgb_anomaly)
            flow_anomaly = self.mlp_flow(flow_anomaly)

            # normalise the features
            rgb_anomaly = F.normalize(rgb_anomaly, p=2, dim=1)
            flow_anomaly = F.normalize(flow_anomaly, p=2, dim=1)
            anomaly_inputs = torch.cat([rgb_anomaly, flow_anomaly], dim = 2)
            anomaly_inputs = self.t_model(anomaly_inputs)


            inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1)
            batch_size = inputs.shape[0]
            inputs = inputs.view(-1, inputs.size(-1))
            outputs = self.mil_model(inputs)
            loss_value = self.criterion(outputs, batch_size, self.device)
            return loss_value
        else:
            rgb_anomaly, flow_anomaly = anomaly_inputs
            rgb_anomaly, flow_anomaly = rgb_anomaly.to(self.device), flow_anomaly.to(self.device)
            # rgb_anomaly = torch.cat([rgb_anomaly, patch_anomaly], dim = -1) 
            rgb_anomaly = self.mlp_rgb(rgb_anomaly)
            flow_anomaly = self.mlp_flow(flow_anomaly)

            # normalise the features
            rgb_anomaly = F.normalize(rgb_anomaly, p=2, dim=1)
            flow_anomaly = F.normalize(flow_anomaly, p=2, dim=1)
            anomaly_inputs = torch.cat([rgb_anomaly, flow_anomaly], dim = 2)
            inputs = self.t_model(anomaly_inputs)
            inputs = inputs.view(-1, inputs.size(-1))
            outputs = self.mil_model(inputs)
            return outputs



def train(epoch, model, normal_train_loader, anomaly_train_loader, optimizer):
    print('\nEpoch: %d' % epoch)
    model.train()

    train_loss = 0
    correct = 0
    total = 0
    loader = zip(normal_train_loader, anomaly_train_loader)
    pbar = tqdm(enumerate(loader, 1))
    for batch_idx, (normal_inputs, anomaly_inputs) in pbar:
        loss = model(anomaly_inputs, normal_inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # print(f"Processed [{batch_idx}]/[{min(len(normal_train_loader), len(anomaly_train_loader))}] MIL Loss : {loss.item():.4f}")
        pbar.set_description(f"Processed [{batch_idx}]/[{min(len(normal_train_loader), len(anomaly_train_loader))}] MIL Loss : {loss.item():.4f}")
        
    pbar.clear()
    pbar.write('loss = {}'.format(train_loss/len(normal_train_loader)))

    return train_loss/len(normal_train_loader)

def save_predictions(epoch, model, anomaly_pred_loader, normal_pred_loader):
    model.eval()
    pred_dict = {}
    with torch.no_grad():
        for i, (data) in enumerate(anomaly_pred_loader):
            inputs = data[:2]
            video_name = data[2][0].split(os.path.sep)[-1]
            score = model(inputs)
            score = score.cpu().detach().numpy().reshape(-1).tolist()
            pred_dict[video_name] = score

        for i, (data) in enumerate(normal_pred_loader):
            inputs = data[:2]
            video_name = data[2][0].split(os.path.sep)[-1]
            score = model(inputs)
            score = score.cpu().detach().numpy().reshape(-1).tolist()
            pred_dict[video_name] = score
    
    return pred_dict

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


    fpr, tpr, thresholds=metrics.roc_curve(all_gt_list,all_score_list,pos_label=1)
    auc=metrics.auc(fpr,tpr)
    return auc


def main() -> None:
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

        normal_train_dataset = Normal_Loader(rgb_models,is_train=1)
        normal_test_dataset = Normal_Loader(rgb_models, is_train=0)

        anomaly_train_dataset = Anomaly_Loader(rgb_models, is_train=1)
        anomaly_test_dataset = Anomaly_Loader(rgb_models, is_train=0)
        TRAIN_BATCH_SIZE = 30
        VALID_BATCH_SIZE = 1
        normal_train_loader = DataLoader(normal_train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers= 8, drop_last = True)
        normal_pred_loader = DataLoader(normal_train_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=4)
        normal_test_loader = DataLoader(normal_test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=4)

        anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers= 8, drop_last = True) 
        anomaly_pred_loader = DataLoader(anomaly_train_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=4)
        anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=4)

        device = 'cuda:7'
        mlp_rgb = MLP(input_dim=mlp_size, hidden_dim=mlp_hidden_dim, output_dim=mlp_output_dim)
        mlp_flow = MLP(input_dim=1024, hidden_dim=512, output_dim=512)

        mil_model = ModifiedMILModel(input_dim = 1024, hidden_dim1 = 512)

        t_model = LSTMDisentangledAttention()

        criterion = MIL
        model = MILModel(mlp_rgb, mlp_flow, mil_model, t_model,criterion, device).to(device)

        params = [
            {'params': model.mlp_rgb.parameters(), 'lr': 1e-3},
            {'params': model.mlp_flow.parameters(), 'lr': 1e-3},
            {'params' : model.mil_model.parameters()},
            {'params' : model.t_model.parameters(), 'lr' : 1e-4}
        ]
        optimizer = torch.optim.Adagrad(params , lr= 1e-3, weight_decay=0.001)
        
        best_auc = 0.5
        auc_scores = []
        n_epochs = 100
        pbar_main = tqdm(range(0, n_epochs))
        for epoch in pbar_main:
            epoch_train_loss = train(epoch, model, normal_train_loader, anomaly_train_loader, optimizer)
            pred_dict = save_predictions(epoch, model, anomaly_pred_loader, normal_pred_loader)
            epoch_auc = test(epoch, model, anomaly_test_loader, normal_test_loader)
            auc_scores.append(epoch_auc)
            if epoch_auc > best_auc:
                best_auc = epoch_auc
                best_pred_dict = pred_dict
            pbar_main.write(f"The AUC SCORE is {epoch_auc}")
            pbar_main.write(f"The BEST AUC SCORE for MIL Model is {best_auc}")
        
        save_file_name = "_".join([model.model_name for model in rgb_models]) + ".pickle"
        # with open(save_file_name, "wb") as f:
        #     pickle.dump(best_pred_dict, f)
        

if __name__ == "__main__":
    #python main_backbones.py --multirun t_model=gru,rnn,lstm,att,gru_att,dis_att,gru_dis_att,rnn_dis_att,tconv,rtfm
    main()