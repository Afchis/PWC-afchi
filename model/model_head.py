import torch
import torch.nn as nn

# import class()
from model_parts import Encoder, Decoder


img1 = torch.rand([2, 3, 512, 512], requires_grad=True).cuda()
img2 = torch.rand([2, 3, 512, 512], requires_grad=True).cuda()


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class ModelHead(nn.Module):
    def __init__(self):
        super(ModelHead, self).__init__()
        self.encoder = Encoder()

        self.decoder_64 = Decoder(in_channels=0)
        self.deconv_feat_64 = deconv(in_planes=529, out_planes=2)
        self.deconv_pred_64 = deconv(in_planes=2, out_planes=2)

        self.decoder_32 = Decoder(in_channels=128+2+2)
        self.deconv_feat_32 = deconv(in_planes=661, out_planes=2)
        self.deconv_pred_32 = deconv(in_planes=2, out_planes=2)

        self.decoder_16 = Decoder(in_channels=96+2+2)
        self.deconv_feat_16 = deconv(in_planes=629, out_planes=2)
        self.deconv_pred_16 = deconv(in_planes=2, out_planes=2)

        self.decoder_8 = Decoder(in_channels=64+2+2)
        self.deconv_feat_8 = deconv(in_planes=597, out_planes=2)
        self.deconv_pred_8 = deconv(in_planes=2, out_planes=2)

        self.decoder_4 = Decoder(in_channels=32+2+2)
        self.deconv_feat_4 = deconv(in_planes=565, out_planes=2)
        self.deconv_pred_4 = deconv(in_planes=2, out_planes=2)

        self.decoder_2 = Decoder(in_channels=16+2+2)

    def forward(self, img1, img2):
        feats_1 = self.encoder(img1)
        feats_2 = self.encoder(img2)

        flow_feat_64, flow_pred_64 = self.decoder_64(feats_1[-1], feats_2[-1], flow_feat=None, flow_pred=None)
        flow_feat_64 = self.deconv_feat_64(flow_feat_64)
        flow_pred_64 = self.deconv_pred_64(flow_pred_64)

        flow_feat_32, flow_pred_32 = self.decoder_32(feats_1[-2], feats_2[-2], flow_feat_64, flow_pred_64)
        flow_feat_32 = self.deconv_feat_32(flow_feat_32)
        flow_pred_32 = self.deconv_pred_32(flow_pred_32)

        flow_feat_16, flow_pred_16 = self.decoder_16(feats_1[-3], feats_2[-3], flow_feat_32, flow_pred_32)
        flow_feat_16 = self.deconv_feat_16(flow_feat_16)
        flow_pred_16 = self.deconv_pred_16(flow_pred_16)

        flow_feat_8, flow_pred_8 = self.decoder_8(feats_1[-4], feats_2[-4], flow_feat_16, flow_pred_16)
        flow_feat_8 = self.deconv_feat_8(flow_feat_8)
        flow_pred_8 = self.deconv_pred_8(flow_pred_8)

        flow_feat_4, flow_pred_4 = self.decoder_4(feats_1[-5], feats_2[-5], flow_feat_8, flow_pred_8)
        flow_feat_4 = self.deconv_feat_4(flow_feat_4)
        flow_pred_4 = self.deconv_pred_4(flow_pred_4)

        flow_feat_2, flow_pred_2 = self.decoder_2(feats_1[-6], feats_2[-6], flow_feat_4, flow_pred_4)
        print("done")
        if self.training:
            return flow_pred_64, flow_pred_32, flow_pred_16, flow_pred_8, flow_pred_4, flow_pred_2
        return flow_pred_2


model = ModelHead().cuda()
model(img1, img2).sum().backward()