import argparse 

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# import class()
from model.model_head import ModelHead
from utils.logger import Logger

# import def()
from dataloader.dataloader import Loader
from loss_metric.losses import Loss, EPE
from utils.flow2img import SaveFlowImg


parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="cuda:0", help="Device number: cuda:*")
parser.add_argument("--tb", type=str, default="None", help="Tensorboard name")
parser.add_argument("--dataset", type=str, default="MpiSintel", help="Dataset name")
parser.add_argument("--weights", type=str, default="deafult_weights", help="Weights name")
parser.add_argument("--batch", type=int, default=4, help="Batch size")
parser.add_argument("--num_workers", type=int, default=4, help="Num workers")
parser.add_argument("--epochs", type=int, default=1000, help="Num epochs")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
parser.add_argument("--vis", type=bool, default=False, help="Visual prediction")

args = parser.parse_args()


# init tensorboard: !tensorboard --logdir=ignore/runs
writer = SummaryWriter('ignore/runs')
print("Tensorboard name: %s" % args.tb)


# init model
device = torch.device(args.device)
model = ModelHead().to(device)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Params: %s" % params)
try:
    model.load_state_dict(torch.load("ignore/weights/%s.pth" % args.weights), strict=False)
    print("Load weights: %s.pth" % args.weights)
except FileNotFoundError:
    print("Create new weights: %s.pth" % args.weights)
    pass

def save_model(i, iter):
    if i % iter == 0 and i != 0:
        torch.save(model.state_dict(), "ignore/weights/%s.pth" % args.weights)


# init dataloader
train_loader, valid_loader = Loader(dataset=args.dataset, batch_size=args.batch, num_workers=args.num_workers, shuffle=True)
print("Batch size:", args.batch)
print("Train data:", len(train_loader)*args.batch)
print("Valid data:", len(valid_loader)*args.batch)


# init optimizer and lr scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400000, 600000, 800000, 1000000], gamma=0.5)


def train():
    logger = Logger(len_train=len(train_loader), len_valid=len(valid_loader))
    for epoch in range(args.epochs):
        logger.init()
        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            img1, img2, flow_label = data
            img1, img2, flow_label = img1.to(device), img2.to(device), flow_label.to(device)
            flow_preds = model(img1, img2)
            loss = Loss(flow_preds, flow_label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            metric = EPE(flow_preds, flow_label)
            logger.update("train_iter", None)
            logger.update("train_loss", loss.detach().item())
            logger.update("train_metric", metric)            
            logger.printer_train()
            logger.visual_train(vis=args.vis, Visual=SaveFlowImg, pred=flow_preds, label=flow_label)
            logger.tensorboard_iter(writer=writer, tb=args.tb)
            save_model(i, iter=200)

        for i, data in enumerate(valid_loader):
            img1, img2, flow_label = data
            img1, img2, flow_label = img1.to(device), img2.to(device), flow_label.to(device)
            flow_preds = model(img1, img2)
            loss = Loss(flow_preds, flow_label)
            metric = EPE(flow_preds, flow_label)
            logger.update("valid_iter", None)
            logger.update("valid_loss", loss.detach().item())
            logger.update("valid_metric", metric)
            logger.printer_valid()
        logger.tensorboard_epoch(writer=writer, tb=args.tb)
        logger.printer_epoch()
        torch.save(model.state_dict(), "ignore/weights/checkpoint_e%03d.pth" % (epoch+1))




if __name__ == "__main__":
    train()

