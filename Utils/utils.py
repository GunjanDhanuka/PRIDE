import numpy as np
import torch
import os
import random


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)

    r = np.linspace(0, len(feat), length + 1, dtype=np.int)
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i] : r[i + 1], :], 0)
        else:
            new_feat[i, :] = feat[r[i], :]
    return new_feat


def seg2frame(actual_frames, score, gt, fixed_len=True):
    detection_score_per_frame = np.zeros(actual_frames)
    if not fixed_len:
        for i in range((actual_frames - 1) // 16 + 1):
            detection_score_per_frame[i * 16 : (i + 1) * 16] = score[i]
    else:
        thirty2_shots = np.round(np.linspace(0, actual_frames // 16, 33))
        for i in range(len(thirty2_shots) - 1):
            ss = int(thirty2_shots[i])
            ee = int(thirty2_shots[i + 1]) - 1

            if ee < ss:
                detection_score_per_frame[ss * 16 : (ss + 1) * 16] = score[i]
            else:
                detection_score_per_frame[ss * 16 : (ee + 1) * 16] = score[i]

    gt_by_frame = np.zeros(actual_frames)

    # check index start 0 or 1
    for i in range(len(gt) // 2):
        s = gt[i * 2]
        e = min(gt[i * 2 + 1], actual_frames)
        gt_by_frame[s - 1 : e] = 1
    return detection_score_per_frame, gt_by_frame


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
