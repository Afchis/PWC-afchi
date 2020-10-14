import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# import def()
from .correlation import FunctionCorrelation


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_relu_2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
        self.conv_relu_4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
        self.conv_relu_8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
        self.conv_relu_16 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
        self.conv_relu_32 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
        self.conv_relu_64 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

    def forward(self, x):
        f2 = self.conv_relu_2(x)
        f4 = self.conv_relu_4(f2)
        f8 = self.conv_relu_8(f4)
        f16 = self.conv_relu_16(f8)
        f32 = self.conv_relu_32(f16)
        f64 = self.conv_relu_64(f32)
        return f2, f4, f8, f16, f32, f64


class Decoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = 81 + in_channels
        self.out_ch_list = [128, 128, 96, 64, 32, 2]
        self.in_ch_list = np.cumsum([self.in_channels] + self.out_ch_list)
        self.pred_layer_0 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_ch_list[0], out_channels=self.out_ch_list[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
        self.pred_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_ch_list[1], out_channels=self.out_ch_list[1], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
        self.pred_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_ch_list[2], out_channels=self.out_ch_list[2], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
        self.pred_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_ch_list[3], out_channels=self.out_ch_list[3], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
        self.pred_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_ch_list[4], out_channels=self.out_ch_list[4], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
        self.pred_layer_5 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_ch_list[5], out_channels=self.out_ch_list[5], kernel_size=3, stride=1, padding=1),
            )

    def _warp(self, x, flo):
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.to(device=x.device)
        vgrid = torch.autograd.Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid, align_corners=False)
        mask = torch.autograd.Variable(torch.ones(x.size())).to(device=x.device)
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=False)

        # if W==128:
            # np.save('mask.npy', mask.cpu().data.numpy())
            # np.save('warp.npy', output.cpu().data.numpy())
        
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask

    def _corr_relu(self, x1, x2):
        return F.leaky_relu(input=FunctionCorrelation(tenFirst=x1, tenSecond=x2), negative_slope=0.1, inplace=False)

    def forward(self, feat1, feat2, flow_feat, flow_pred):
        if flow_pred is not None:
            feat2 = self._warp(feat2, flow_pred)
            flow_feat = torch.cat([self._corr_relu(feat1, feat2), feat1, flow_feat, flow_pred], dim=1)
        else:
            flow_feat = self._corr_relu(feat1, feat2)
        flow_feat = torch.cat([flow_feat, self.pred_layer_0(flow_feat)], dim=1)
        flow_feat = torch.cat([flow_feat, self.pred_layer_1(flow_feat)], dim=1)
        flow_feat = torch.cat([flow_feat, self.pred_layer_2(flow_feat)], dim=1)
        flow_feat = torch.cat([flow_feat, self.pred_layer_3(flow_feat)], dim=1)
        flow_feat = torch.cat([flow_feat, self.pred_layer_4(flow_feat)], dim=1)
        flow_pred = self.pred_layer_5(flow_feat)
        return flow_feat, flow_pred


class Refiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.refiner = nn.Sequential(
            nn.Conv2d(in_channels=565, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
            )

    def forward(self, pred_feat, pred_flow):
        return self.refiner(pred_feat) + pred_flow

