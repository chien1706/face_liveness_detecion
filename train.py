import torch
import torch.nn as nn
from data.dataset import FLDDataset
from torch.utils.data import DataLoader
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torch import optim as optim
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/1_frame_resnet50")
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50(weights= ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.fc = torch.nn.Linear(2048, 1024)
        self.relu = nn.ReLU()
        self.fc2 = torch.nn.Linear(1024, 1)
        print(self.resnet)
    def forward(self, x):
        x = self.resnet(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.head(x)
        return x.sigmoid()
model = Model()
dataset = FLDDataset("/home/aimenext/luantt/zaloai/face_liveness_detection/dataset/train")
train_loader = DataLoader(dataset, batch_size = 16, shuffle=True)
epochs = 100
device = torch.device("cuda")
model.to(device)
criterion = nn.BCELoss()
optim = optim.Adam(model.parameters(), lr = 1e-3)
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.
    for i, (label, image) in enumerate(train_loader):
        optim.zero_grad()
        label = label.to(device)
        image = image.to(device)
        outputs = model(image)
        # output shape []
        label = label.unsqueeze(1)
        loss = criterion(outputs, label)
        loss.backward()
        optim.step()
        epoch_loss += loss.item()
        print("Epoch {}, Step [{}/{}], Loss [{}]".format(epoch, i, len(train_loader), loss.item()))

    writer.add_scalar("Epoch_loss", epoch_loss / len(train_loader), epoch)        
    if epoch % 10 == 0:
        torch.save(model.state_dict(), "weights/resnet_1f_epoch_{}.pth".format(epoch))
    