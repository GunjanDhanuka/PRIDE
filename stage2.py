import sys
sys.path.append('..')
from torch.utils.data import DataLoader, Dataset
import os
from sklearn import metrics
from PRIDE.Models.disentangled_attention import LSTMDisentangledAttention
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import NamedTuple
import pickle

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


import warnings
warnings.simplefilter("ignore")

#Setting the seed
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(40)

def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
    
    r = np.linspace(0, len(feat), length+1, dtype=np.int)
    for i in range(length):
        if r[i]!=r[i+1]:
            new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
        else:
            new_feat[i,:] = feat[r[i],:]
    return new_feat

def MIL(y_pred, batch_size, device, is_transformer=0):
    loss = torch.tensor(0.).to(device)
    loss_intra = torch.tensor(0.).to(device)
    sparsity = torch.tensor(0.).to(device)
    smooth = torch.tensor(0.).to(device)
    if is_transformer==0:
        y_pred = y_pred.view(batch_size, -1)
    else:
        y_pred = torch.sigmoid(y_pred)

    for i in range(batch_size):

        y_anomaly = y_pred[i, :32]
        y_normal  = y_pred[i, 32:]

        y_anomaly_max = torch.max(y_anomaly) # anomaly
        y_normal_max = torch.max(y_normal) # normal


        loss += F.relu(1.-y_anomaly_max+y_normal_max)


        sparsity += torch.sum(y_anomaly)*0.00008
        smooth += torch.sum((y_pred[i,:31] - y_pred[i,1:32])**2)*0.00008
    loss = (loss+sparsity+smooth)/batch_size

    return loss

def Binloss(y_pred, batch_size, scores_anomaly, scores_normal, device, is_transformer=0):

    y_pred = y_pred.view(batch_size, -1) 
    y_anomaly = y_pred[:,:32]
    y_normal = y_pred[:,32:]
    y_anomaly_pred = y_anomaly.contiguous().view(-1)
    y_normal_pred = y_normal.contiguous().view(-1)
    y_anomaly_gt = scores_anomaly.contiguous().view(-1)
    y_normal_gt = scores_normal.contiguous().view(-1)

    assert len(y_normal_pred) == len(y_anomaly_pred) == len(y_anomaly_gt) == len(y_normal_gt), "Debug pseudo labels"

    y_pred = torch.cat([y_anomaly_pred, y_normal_pred], dim  = 0)
    y_gt = torch.cat([y_anomaly_gt, y_normal_gt], dim  = 0)

    loss = F.binary_cross_entropy(y_pred, y_gt)
    print(loss.shape)
    sparsity = torch.sum(y_anomaly, dim = -1).mean() * 0.00008
    smooth = torch.sum((y_anomaly[:,:31] - y_anomaly[:,1:32]) ** 2, dim = -1).mean() * 0.00008

    loss  = (loss + sparsity + smooth)

    
    return loss

def Focalloss(y_pred, batch_size, scores_anomaly, scores_normal, device, is_transformer=0, alpha=0.25, gamma=2):

    y_pred = y_pred.view(batch_size, -1) 
    y_anomaly = y_pred[:,:32]
    y_normal = y_pred[:,32:]
    y_anomaly_pred = y_anomaly.contiguous().view(-1)
    y_normal_pred = y_normal.contiguous().view(-1)
    y_anomaly_gt = scores_anomaly.contiguous().view(-1)
    y_normal_gt = scores_normal.contiguous().view(-1)

    assert len(y_normal_pred) == len(y_anomaly_pred) == len(y_anomaly_gt) == len(y_normal_gt), "Debug pseudo labels"

    y_pred = torch.cat([y_anomaly_pred, y_normal_pred], dim  = 0)
    y_gt = torch.cat([y_anomaly_gt, y_normal_gt], dim  = 0)

    alpha = torch.tensor([alpha, 1-alpha]).to(device)

    BCE_loss = F.binary_cross_entropy(y_pred, y_gt, reduction = "none")
    # y_gt = y_gt.type(torch.long)
    y_gt = y_gt.round().type(torch.long)
    at = alpha.gather(0, y_gt.data.view(-1))
    pt = torch.exp(-BCE_loss)
    F_loss = at*(1-pt)**gamma * BCE_loss
    loss = F_loss.mean()
    sparsity = torch.sum(y_anomaly, dim = -1).mean() * 0.00008
    smooth = torch.sum((y_anomaly[:,:31] - y_anomaly[:,1:32]) ** 2, dim = -1).mean() * 0.00008

    loss  = (loss + sparsity + smooth)
    return loss

def KLDloss(y_pred, batch_size, scores_anomaly, scores_normal, device, is_transformer=0, alpha=0.25, gamma=2):

    y_pred = y_pred.view(batch_size, -1) 
    y_anomaly = y_pred[:,:32]
    y_normal = y_pred[:,32:]
    y_anomaly_pred = y_anomaly.contiguous().view(-1)
    y_normal_pred = y_normal.contiguous().view(-1)
    y_anomaly_gt = scores_anomaly.contiguous().view(-1)
    y_normal_gt = scores_normal.contiguous().view(-1)

    assert len(y_normal_pred) == len(y_anomaly_pred) == len(y_anomaly_gt) == len(y_normal_gt), "Debug pseudo labels"

    y_pred = torch.cat([y_anomaly_pred, y_normal_pred], dim  = 0)
    y_gt = torch.cat([y_anomaly_gt, y_normal_gt], dim  = 0)

    loss = F.kl_div(y_pred, y_gt, reduction = "none")
    sparsity = torch.sum(y_anomaly, dim = -1).mean() * 0.00008
    smooth = torch.sum((y_anomaly[:,:31] - y_anomaly[:,1:32]) ** 2, dim = -1).mean() * 0.00008
    print(loss.shape)
    loss  = (loss + sparsity + smooth)
    return loss

def preprocess_score(scores_list, k = 3, average = True):
    if average:
        average_filter = np.ones(k)
        average_filter /= k
        pad_scores = [0] * (k//2) + scores_list + [0] * (k//2)
        avg_scores = np.convolve(pad_scores, average_filter, "valid")
    else:
        avg_scores = np.array(scores_list)
    min_val, max_val = avg_scores.min(), avg_scores.max()
    avg_scores = (avg_scores - min_val)/(max_val - min_val)
    return avg_scores


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
            data_list = 'files/train_normal.txt'
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = 'files/test_normalv2.txt'
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
            # self.data_list = self.data_list[:-10]
        with open("i3d_swin_s3d.pickle", "rb") as f:
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
                features = np.load(os.path.join(datapath, self.data_list[idx][:-1]+'.npy'))
                if req_process:
                    features = process_feat(features, 32)
                rgb_npy.append(features)
            rgb_npy = np.concatenate(rgb_npy, axis=-1)
            flow_npy = np.load(os.path.join(self.flow_dir, self.data_list[idx][:-1]+'.npy'))
            # patch_npy = np.load(os.path.join(self.vit_patch_dir, self.data_list[idx][:-1]+'.npy'))
            # scores = torch.tensor([0] * 32, dtype = torch.float32)
            video_name = self.data_list[idx][:-1].split(os.path.sep)[-1]
            # scores = torch.tensor(self.pseudo_labels[video_name], dtype = torch.float32)
            scores = torch.tensor([0] * 32, dtype = torch.float32)
            stack_npy = rgb_npy, flow_npy, scores
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
            data_list = 'files/train_anomaly.txt'
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = 'files/test_anomalyv2.txt'
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()

        with open("i3d_swin_s3d.pickle", "rb") as f:
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
                features = np.load(os.path.join(datapath, self.data_list[idx][:-1]+'.npy'))
                if req_process:
                    features = process_feat(features, 32)
                rgb_npy.append(features)
            rgb_npy = np.concatenate(rgb_npy, axis=-1)
            flow_npy = np.load(os.path.join(self.flow_dir, self.data_list[idx][:-1]+'.npy'))
            # patch_npy = np.load(os.path.join(self.vit_patch_dir, self.data_list[idx][:-1]+'.npy'))
            video_name = self.data_list[idx][:-1].split(os.path.sep)[-1]
            scores = torch.tensor(preprocess_score(self.pseudo_labels[video_name], average=False), dtype = torch.float32)
            stack_npy = rgb_npy, flow_npy, scores
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

def seg2frame(actual_frames, score, gt, fixed_len=True):
    detection_score_per_frame = np.zeros(actual_frames)
    if not fixed_len:
        for i in range((actual_frames-1)//16+1):
            detection_score_per_frame[i*16:(i+1)*16] = score[i]
    else:
        thirty2_shots = np.round(np.linspace(0, actual_frames//16, 33))
        for i in range(len(thirty2_shots)-1):
            ss = int(thirty2_shots[i])
            ee = int(thirty2_shots[i+1])-1

            if ee<ss:
                detection_score_per_frame[ss*16:(ss+1)*16] = score[i]
            else:
                detection_score_per_frame[ss*16:(ee+1)*16] = score[i]

    gt_by_frame = np.zeros(actual_frames)

    # check index start 0 or 1
    for i in range(len(gt)//2):
        s = gt[i*2]
        e = min(gt[i*2+1], actual_frames)
        gt_by_frame[s-1:e] = 1
    return detection_score_per_frame, gt_by_frame


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
            rgb_normal, flow_normal, scores_normal = normal_inputs
            rgb_normal, flow_normal, scores_normal = rgb_normal.to(self.device), flow_normal.to(self.device), scores_normal.to(self.device)
            # rgb_patch_normal = torch.cat([rgb_normal, patch_normal], dim = -1) 
            rgb_normal = self.mlp_rgb(rgb_normal)
            flow_normal = self.mlp_flow(flow_normal)

            # normalise the features
            rgb_normal = F.normalize(rgb_normal, p=2, dim=1)
            flow_normal = F.normalize(flow_normal, p=2, dim=1)
            normal_inputs = torch.cat([rgb_normal, flow_normal], dim = 2)
            normal_inputs = self.t_model(normal_inputs)

            rgb_anomaly, flow_anomaly, scores_anomaly = anomaly_inputs
            rgb_anomaly, flow_anomaly, scores_anomaly = rgb_anomaly.to(self.device), flow_anomaly.to(self.device), scores_anomaly.to(self.device)
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
            loss_value = self.criterion(outputs, batch_size, scores_anomaly, scores_normal, self.device)
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
    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(loader, 1):
        loss = model(anomaly_inputs, normal_inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print(f"Processed [{batch_idx}]/[{min(len(normal_train_loader), len(anomaly_train_loader))}] MIL Loss : {loss.item():.4f}")
        
    print('loss = {}'.format(train_loss/len(normal_train_loader)))

    return train_loss/len(normal_train_loader)


def test(epoch, model, anomaly_test_loader, normal_test_loader):
    model.eval()
    auc = 0
    all_gt_list = []
    all_score_list = []
    pred_dict = {}
    with torch.no_grad():
        for i, (data) in enumerate(anomaly_test_loader):
            inputs, gts, frames, name = data
            score = model(inputs)
            score = score.cpu().detach().numpy()
            score_list, gt_list = seg2frame(frames.item(), score, gts)
            all_gt_list.extend(gt_list)
            all_score_list.extend(score_list)

            video_name = name[0].split(os.path.sep)[-1]
            pred_dict[video_name] = score_list.tolist()


        for i, (data2) in enumerate(normal_test_loader):
            inputs2, gts2, frames2, name2 = data2
            score2 = model(inputs2)
            score2 = score2.cpu().detach().numpy()
            score_list2, gt_list2 = seg2frame(frames2.item(), score2, [])
            all_gt_list.extend(gt_list2.tolist())
            all_score_list.extend(score_list2.tolist())

            video_name = name2[0].split(os.path.sep)[-1]
            pred_dict[video_name] = score_list2.tolist()


    fpr, tpr, thresholds=metrics.roc_curve(all_gt_list,all_score_list,pos_label=1)
    auc=metrics.auc(fpr,tpr)
    return auc, pred_dict


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
        TRAIN_BATCH_SIZE = 4
        VALID_BATCH_SIZE = 1
        normal_train_loader = DataLoader(normal_train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers= 8, drop_last = True)
        normal_test_loader = DataLoader(normal_test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=4)

        anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers= 8, drop_last = True) 
        anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=4)

        device = 'cuda:7'
        mlp_rgb = MLP(input_dim=mlp_size, hidden_dim=mlp_hidden_dim, output_dim=mlp_output_dim)
        mlp_flow = MLP(input_dim=1024, hidden_dim=512, output_dim=512)

        mil_model = ModifiedMILModel(input_dim = 1024, hidden_dim1 = 512)

        t_model = LSTMDisentangledAttention()

        criterion = Binloss
        model = MILModel(mlp_rgb, mlp_flow, mil_model, t_model,criterion, device).to(device)

        params = [
            {'params': model.mlp_rgb.parameters(), 'lr': 1e-3},
            {'params': model.mlp_flow.parameters(), 'lr': 1e-3},
            {'params' : model.mil_model.parameters()},
            {'params' : model.t_model.parameters(), 'lr' : 1e-4}
        ]
        optimizer = torch.optim.Adagrad(params , lr= 1e-3, weight_decay=0.001)

        # optimizer = torch.optim.AdamW(params , lr= 1e-3, weight_decay=0.001)
        
        best_auc = 0.5
        auc_scores = []
        n_epochs = 2
        for epoch in range(0, n_epochs):
            epoch_train_loss = train(epoch, model, normal_train_loader, anomaly_train_loader, optimizer)
            epoch_auc, pred_dict = test(epoch, model, anomaly_test_loader, normal_test_loader)
            auc_scores.append(epoch_auc)
            if epoch_auc > best_auc:
                best_auc = epoch_auc
                best_pred_dict = pred_dict
            print(f"The AUC SCORE is {epoch_auc:.4f}")
            print(f"The BEST AUC SCORE for MIL Model is {best_auc:.4f}")
        plt.plot(auc_scores)
        plt.xlabel('Epochs')
        plt.ylabel('AUC Score')
        plt.title(f'{criterion.__name__}, Best AUC = {best_auc}')
        # plt.savefig("focal_corr_noavg.png", dpi=300)
        

if __name__ == "__main__":
    #python main_backbones.py --multirun t_model=gru,rnn,lstm,att,gru_att,dis_att,gru_dis_att,rnn_dis_att,tconv,rtfm
    main()