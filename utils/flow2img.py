import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import flowiz as fz

UpScale = nn.Upsample(scale_factor=2, mode='nearest')

def SaveFlowImg(flows_pred, flow_label):
    flow_label = F.avg_pool2d(flow_label, kernel_size=2)
    flow_pred_64, flow_pred_32, flow_pred_16, flow_pred_8, flow_pred_4 = flows_pred
    flow_pred_4 = UpScale(flow_pred_4)
    flow_pred_8 = UpScale(UpScale(flow_pred_8))
    flow_pred_16 = UpScale(UpScale(UpScale(flow_pred_16)))
    flow_pred_32 = UpScale(UpScale(UpScale(UpScale(flow_pred_32))))
    flow_pred_64 = UpScale(UpScale(UpScale(UpScale(UpScale(flow_pred_64)))))
    for batch in range(flow_label.size(0)):
        flow = fz.convert_from_flow(flow_pred_4[batch].permute(1, 2, 0).detach().cpu().numpy())
        flow = Image.fromarray(flow)
        flow.save("ignore/visual/%d_pred_4.png" % batch)
        flow = fz.convert_from_flow(flow_pred_8[batch].permute(1, 2, 0).detach().cpu().numpy())
        flow = Image.fromarray(flow)
        flow.save("ignore/visual/%d_pred_8.png" % batch)
        flow = fz.convert_from_flow(flow_pred_16[batch].permute(1, 2, 0).detach().cpu().numpy())
        flow = Image.fromarray(flow)
        flow.save("ignore/visual/%d_pred_16.png" % batch)
        flow = fz.convert_from_flow(flow_pred_32[batch].permute(1, 2, 0).detach().cpu().numpy())
        flow = Image.fromarray(flow)
        flow.save("ignore/visual/%d_pred_32.png" % batch)
        flow = fz.convert_from_flow(flow_pred_64[batch].permute(1, 2, 0).detach().cpu().numpy())
        flow = Image.fromarray(flow)
        flow.save("ignore/visual/%d_pred_64.png" % batch)
        flow = fz.convert_from_flow(flow_label[batch].permute(1, 2, 0).detach().cpu().numpy())
        flow = Image.fromarray(flow)
        flow.save("ignore/visual/%d_label.png" % batch)

