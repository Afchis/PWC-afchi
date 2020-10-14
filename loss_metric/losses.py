import torch
import torch.nn.functional as F


def Loss(flow_preds, flow_label, loss="L2"):
    flow_pred_64, flow_pred_32, flow_pred_16, flow_pred_8, flow_pred_4 = flow_preds
    flow_label_2 = F.avg_pool2d(flow_label*20, kernel_size=2)
    flow_label_4 = F.avg_pool2d(flow_label_2, kernel_size=2)
    flow_label_8 = F.avg_pool2d(flow_label_4, kernel_size=2)
    flow_label_16 = F.avg_pool2d(flow_label_8, kernel_size=2)
    flow_label_32 = F.avg_pool2d(flow_label_16, kernel_size=2)
    flow_label_64 = F.avg_pool2d(flow_label_32, kernel_size=2)
    loss = 0.32*F.mse_loss(flow_pred_64, flow_label_64, reduction='mean') + 0.08*F.mse_loss(flow_pred_32, flow_label_32, reduction='mean') + \
           0.02*F.mse_loss(flow_pred_16, flow_label_16, reduction='mean') + 0.01*F.mse_loss(flow_pred_8, flow_label_8, reduction='mean') + \
           0.005*F.mse_loss(flow_pred_4, flow_label_4, reduction='mean')
    return loss


def EPE(flow_preds, flow_label):
    _, _, _, _, flow_pred_4 = flow_preds
    flow_label_2 = F.avg_pool2d(flow_label, kernel_size=2)
    flow_label_4 = F.avg_pool2d(flow_label_2, kernel_size=2)
    return torch.norm(flow_label_4*5-flow_pred_4, p=2, dim=1).mean().detach().item()
