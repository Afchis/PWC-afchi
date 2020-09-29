import numpy as np
import torch

from PIL import Image
import flowiz as fz

def SaveFlowImg(flows_pred, flow_label):
    flow_pred_64, flow_pred_32, flow_pred_16, flow_pred_8, flow_pred_4, flow_pred_2 = flows_pred

    flow = fz.convert_from_flow(flow_pred_2[0].permute(1, 2, 0).detach().cpu().numpy())
    flow = Image.fromarray(flow)
    flow.save("flow10.png")
    print("save")

    flow = fz.convert_from_flow(flow_label[0].permute(1, 2, 0).detach().cpu().numpy())
    flow = Image.fromarray(flow)
    flow.save("flow11.png")
    print("save")

    flow = fz.convert_from_flow(flow_pred_4[1].permute(1, 2, 0).detach().cpu().numpy())
    flow = Image.fromarray(flow)
    flow.save("flow20.png")
    print("save")

    flow = fz.convert_from_flow(flow_label[1].permute(1, 2, 0).detach().cpu().numpy())
    flow = Image.fromarray(flow)
    flow.save("flow21.png")
    print("save")

    flow = fz.convert_from_flow(flow_pred_2[2].permute(1, 2, 0).detach().cpu().numpy())
    flow = Image.fromarray(flow)
    flow.save("flow30.png")
    print("save")

    flow = fz.convert_from_flow(flow_label[2].permute(1, 2, 0).detach().cpu().numpy())
    flow = Image.fromarray(flow)
    flow.save("flow31.png")
    print("save")

    flow = fz.convert_from_flow(flow_pred_4[3].permute(1, 2, 0).detach().cpu().numpy())
    flow = Image.fromarray(flow)
    flow.save("flow40.png")
    print("save")

    flow = fz.convert_from_flow(flow_label[3].permute(1, 2, 0).detach().cpu().numpy())
    flow = Image.fromarray(flow)
    flow.save("flow41.png")
    print("save")

    flow = fz.convert_from_flow(flow_pred_2[4].permute(1, 2, 0).detach().cpu().numpy())
    flow = Image.fromarray(flow)
    flow.save("flow50.png")
    print("save")

    flow = fz.convert_from_flow(flow_label[4].permute(1, 2, 0).detach().cpu().numpy())
    flow = Image.fromarray(flow)
    flow.save("flow51.png")
    print("save")

    flow = fz.convert_from_flow(flow_pred_4[5].permute(1, 2, 0).detach().cpu().numpy())
    flow = Image.fromarray(flow)
    flow.save("flow60.png")
    print("save")

    flow = fz.convert_from_flow(flow_label[5].permute(1, 2, 0).detach().cpu().numpy())
    flow = Image.fromarray(flow)
    flow.save("flow61.png")
    print("save")




