import torch

# import class()
from model.model_head import ModelHead

# import def()
from dataloader.dataloader import Loader
from loss_metric.losses import TotalLoss
from utils.flow2img import SaveFlowImg


model = ModelHead().cuda()
train_loader = Loader(batch_size=6, num_workers=12, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

try:
    model.load_state_dict(torch.load('w.pth'))
except FileNotFoundError:
    print("!!!Create new weights!!!: ")
    pass

def train(epochs=1):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i, data in enumerate(train_loader):
            i += 1
            img1, img2, flow_label = data
            img1, img2, flow_label = img1.cuda(), img2.cuda(), flow_label.cuda()
            flows_pred = model(img1, img2)
            loss = TotalLoss(flows_pred, flow_label)
            epoch_loss += loss.detach().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            SaveFlowImg(flows_pred, flow_label)
        print(epoch_loss / i)
        if epoch % 100 == 0 and epoch != 0:
            torch.save(model.state_dict(), 'w.pth')
            print("Save weights")



if __name__ == "__main__":
	train(epochs=100000)