import torch
import torch.nn.functional as F


def TotalLoss(flows_pred, flow_label):
    flow_pred_64, flow_pred_32, flow_pred_16, flow_pred_8, flow_pred_4, flow_pred_2 = flows_pred
    flow_label_2 = F.interpolate(flow_label, size=(192, 256), mode='bilinear', align_corners=False)
    flow_label_4 = F.interpolate(flow_label, size=(96, 128), mode='bilinear', align_corners=False)
    flow_label_8 = F.interpolate(flow_label, size=(48, 64), mode='bilinear', align_corners=False)
    flow_label_16 = F.interpolate(flow_label, size=(24, 32), mode='bilinear', align_corners=False)
    flow_label_32 = F.interpolate(flow_label, size=(12, 16), mode='bilinear', align_corners=False)
    flow_label_64 = F.interpolate(flow_label, size=(6, 8), mode='bilinear', align_corners=False)
    loss = 0.32*F.mse_loss(flow_pred_64, flow_label_64) + 0.32*F.mse_loss(flow_pred_32, flow_label_32) + \
           0.08*F.mse_loss(flow_pred_16, flow_label_16) + 0.02*F.mse_loss(flow_pred_8, flow_label_8) + \
           0.01*F.mse_loss(flow_pred_4, flow_label_4) + 0.005*F.mse_loss(flow_pred_2, flow_label_2)
    return loss