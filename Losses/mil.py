import torch


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
