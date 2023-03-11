import torch
import torch.nn.functional as F

def MIL(y_pred, batch_size, device, is_transformer=0):
    loss = torch.tensor(0.0).to(device)
    loss_intra = torch.tensor(0.0).to(device)
    sparsity = torch.tensor(0.0).to(device)
    smooth = torch.tensor(0.0).to(device)
    if is_transformer == 0:
        y_pred = y_pred.view(batch_size, -1)
    else:
        y_pred = torch.sigmoid(y_pred)

    for i in range(batch_size):
        y_anomaly = y_pred[i, :32]
        y_normal = y_pred[i, 32:]

        y_anomaly_max = torch.max(y_anomaly)  # anomaly
        y_normal_max = torch.max(y_normal)  # normal

        loss += F.relu(1.0 - y_anomaly_max + y_normal_max)

        sparsity += torch.sum(y_anomaly) * 0.00008
        smooth += torch.sum((y_pred[i, :31] - y_pred[i, 1:32]) ** 2) * 0.00008
    loss = (loss + sparsity + smooth) / batch_size

    return loss

def Binloss(
    y_pred, batch_size, scores_anomaly, scores_normal, device, is_transformer=0
):
    y_pred = y_pred.view(batch_size, -1)
    y_anomaly = y_pred[:, :32]
    y_normal = y_pred[:, 32:]
    y_anomaly_pred = y_anomaly.contiguous().view(-1)
    y_normal_pred = y_normal.contiguous().view(-1)
    y_anomaly_gt = scores_anomaly.contiguous().view(-1)
    y_normal_gt = scores_normal.contiguous().view(-1)

    assert (
        len(y_normal_pred)
        == len(y_anomaly_pred)
        == len(y_anomaly_gt)
        == len(y_normal_gt)
    ), "Debug pseudo labels"

    y_pred = torch.cat([y_anomaly_pred, y_normal_pred], dim=0)
    y_gt = torch.cat([y_anomaly_gt, y_normal_gt], dim=0)

    loss = F.binary_cross_entropy(y_pred, y_gt)
    print(loss.shape)
    sparsity = torch.sum(y_anomaly, dim=-1).mean() * 0.00008
    smooth = (
        torch.sum((y_anomaly[:, :31] - y_anomaly[:, 1:32]) ** 2, dim=-1).mean()
        * 0.00008
    )

    loss = loss + sparsity + smooth

    return loss

def Focalloss(
    y_pred,
    batch_size,
    scores_anomaly,
    scores_normal,
    device,
    is_transformer=0,
    alpha=0.25,
    gamma=2,
):
    y_pred = y_pred.view(batch_size, -1)
    y_anomaly = y_pred[:, :32]
    y_normal = y_pred[:, 32:]
    y_anomaly_pred = y_anomaly.contiguous().view(-1)
    y_normal_pred = y_normal.contiguous().view(-1)
    y_anomaly_gt = scores_anomaly.contiguous().view(-1)
    y_normal_gt = scores_normal.contiguous().view(-1)

    assert (
        len(y_normal_pred)
        == len(y_anomaly_pred)
        == len(y_anomaly_gt)
        == len(y_normal_gt)
    ), "Debug pseudo labels"

    y_pred = torch.cat([y_anomaly_pred, y_normal_pred], dim=0)
    y_gt = torch.cat([y_anomaly_gt, y_normal_gt], dim=0)

    alpha = torch.tensor([alpha, 1 - alpha]).to(device)

    BCE_loss = F.binary_cross_entropy(y_pred, y_gt, reduction="none")
    # y_gt = y_gt.type(torch.long)
    y_gt = y_gt.round().type(torch.long)
    at = alpha.gather(0, y_gt.data.view(-1))
    pt = torch.exp(-BCE_loss)
    F_loss = at * (1 - pt) ** gamma * BCE_loss
    loss = F_loss.mean()
    sparsity = torch.sum(y_anomaly, dim=-1).mean() * 0.00008
    smooth = (
        torch.sum((y_anomaly[:, :31] - y_anomaly[:, 1:32]) ** 2, dim=-1).mean()
        * 0.00008
    )

    loss = loss + sparsity + smooth
    return loss


def KLDloss(
    y_pred,
    batch_size,
    scores_anomaly,
    scores_normal,
    device,
    is_transformer=0,
    alpha=0.25,
    gamma=2,
):
    y_pred = y_pred.view(batch_size, -1)
    y_anomaly = y_pred[:, :32]
    y_normal = y_pred[:, 32:]
    y_anomaly_pred = y_anomaly.contiguous().view(-1)
    y_normal_pred = y_normal.contiguous().view(-1)
    y_anomaly_gt = scores_anomaly.contiguous().view(-1)
    y_normal_gt = scores_normal.contiguous().view(-1)

    assert (
        len(y_normal_pred)
        == len(y_anomaly_pred)
        == len(y_anomaly_gt)
        == len(y_normal_gt)
    ), "Debug pseudo labels"

    y_pred = torch.cat([y_anomaly_pred, y_normal_pred], dim=0)
    y_gt = torch.cat([y_anomaly_gt, y_normal_gt], dim=0)

    loss = F.kl_div(y_pred, y_gt, reduction="none")
    sparsity = torch.sum(y_anomaly, dim=-1).mean() * 0.00008
    smooth = (
        torch.sum((y_anomaly[:, :31] - y_anomaly[:, 1:32]) ** 2, dim=-1).mean()
        * 0.00008
    )
    print(loss.shape)
    loss = loss + sparsity + smooth
    return loss