import torch


def Binloss(y_pred, batch_size, device, is_transformer=0):
    targets = torch.tensor([1] * batch_size + [0] * batch_size, dtype=torch.float32).to(
        y_pred.device
    )
    y_pred = y_pred.view(batch_size, -1)
    y_anomaly = y_pred[:, :32]
    y_normal = y_pred[:, 32:]
    y_anomaly_max = torch.max(y_anomaly, dim=-1)[0]
    y_normal_max = torch.max(y_normal, dim=-1)[0]

    loss = F.binary_cross_entropy(
        torch.cat([y_anomaly_max, y_normal_max], dim=0), targets
    )
    sparsity = torch.sum(y_anomaly, dim=-1).mean() * 0.00008
    smooth = (
        torch.sum((y_anomaly[:, :31] - y_anomaly[:, 1:32]) ** 2, dim=-1).mean()
        * 0.00008
    )

    loss = loss + sparsity + smooth

    return loss
