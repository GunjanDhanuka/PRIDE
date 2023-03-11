import torch
import torch.nn as nn
import torch.nn.functional as F


class ModifiedMILModel(nn.Module):
    def __init__(
        self, input_dim=1024, hidden_dim1=512, hidden_dim2=32, output_dim=1, drop_p=0.3
    ):
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
    def __init__(
        self, mlp_rgb, mlp_flow, mil_model, t_model, criterion, device, stage2=False
    ):
        super(MILModel, self).__init__()
        self.mlp_rgb = mlp_rgb
        self.mlp_flow = mlp_flow
        self.mil_model = mil_model
        self.t_model = t_model
        self.criterion = criterion
        self.device = device
        self.stage2 = stage2

    def forward(self, anomaly_inputs, normal_inputs=None):
        if normal_inputs is not None:
            if self.stage2:
                rgb_normal, flow_normal, scores_normal = normal_inputs
                rgb_normal, flow_normal, scores_normal = (
                    rgb_normal.to(self.device),
                    flow_normal.to(self.device),
                    scores_normal.to(self.device),
                )
            else:
                rgb_normal, flow_normal, _ = normal_inputs
                rgb_normal, flow_normal = rgb_normal.to(self.device), flow_normal.to(
                    self.device
                )
            # rgb_patch_normal = torch.cat([rgb_normal, patch_normal], dim = -1)
            rgb_normal = self.mlp_rgb(rgb_normal)
            flow_normal = self.mlp_flow(flow_normal)

            # normalise the features
            rgb_normal = F.normalize(rgb_normal, p=2, dim=1)
            flow_normal = F.normalize(flow_normal, p=2, dim=1)
            normal_inputs = torch.cat([rgb_normal, flow_normal], dim=2)
            normal_inputs = self.t_model(normal_inputs)

            if self.stage2:
                rgb_anomaly, flow_anomaly, scores_anomaly = anomaly_inputs
                rgb_anomaly, flow_anomaly, scores_anomaly = (
                    rgb_anomaly.to(self.device),
                    flow_anomaly.to(self.device),
                    scores_anomaly.to(self.device),
                )
            else:
                rgb_anomaly, flow_anomaly, _ = anomaly_inputs
                rgb_anomaly, flow_anomaly = rgb_anomaly.to(
                    self.device
                ), flow_anomaly.to(self.device)
            # rgb_anomaly = torch.cat([rgb_anomaly, patch_anomaly], dim = -1)
            rgb_anomaly = self.mlp_rgb(rgb_anomaly)
            flow_anomaly = self.mlp_flow(flow_anomaly)

            # normalise the features
            rgb_anomaly = F.normalize(rgb_anomaly, p=2, dim=1)
            flow_anomaly = F.normalize(flow_anomaly, p=2, dim=1)
            anomaly_inputs = torch.cat([rgb_anomaly, flow_anomaly], dim=2)
            anomaly_inputs = self.t_model(anomaly_inputs)

            inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1)
            batch_size = inputs.shape[0]
            inputs = inputs.view(-1, inputs.size(-1))
            outputs = self.mil_model(inputs)
            if self.stage2:
                loss_value = self.criterion(
                    outputs, batch_size, scores_anomaly, scores_normal, self.device
                )
            else:
                loss_value = self.criterion(outputs, batch_size, self.device)
            return loss_value
        else:
            rgb_anomaly, flow_anomaly = anomaly_inputs
            rgb_anomaly, flow_anomaly = rgb_anomaly.to(self.device), flow_anomaly.to(
                self.device
            )
            # rgb_anomaly = torch.cat([rgb_anomaly, patch_anomaly], dim = -1)
            rgb_anomaly = self.mlp_rgb(rgb_anomaly)
            flow_anomaly = self.mlp_flow(flow_anomaly)

            # normalise the features
            rgb_anomaly = F.normalize(rgb_anomaly, p=2, dim=1)
            flow_anomaly = F.normalize(flow_anomaly, p=2, dim=1)
            anomaly_inputs = torch.cat([rgb_anomaly, flow_anomaly], dim=2)
            inputs = self.t_model(anomaly_inputs)
            inputs = inputs.view(-1, inputs.size(-1))
            outputs = self.mil_model(inputs)
            return outputs
