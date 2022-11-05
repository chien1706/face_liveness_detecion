import torch
import torch.nn as nn
from data.test_dataset import FLDTestDataset
from torch.utils.data import DataLoader
import torchvision
from torch import optim as optim
import numpy as np
import glob
from tqdm import tqdm
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()
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
def main():
    model = Model()
    model.load_state_dict(torch.load(pretrained_path))
    model.eval()
    model.to(device)
    result = {}
    dataset = FLDTestDataset(test_path)
    for image, name in tqdm(dataset):
        image = image.to(device)
        image = image.unsqueeze(0)
        output = model(image)
        result[name] = output[0,0].item()
    with open("result/Predict.csv", 'w') as f:
        f.write("fname,liveness_score\n")
        for key, value in result.items():
            f.write("{},{}\n".format(key, value))
        
    

if __name__ == '__main__':
    pretrained_path = "weights/resnet_1f_epoch_30.pth"
    test_path = "dataset/public"
    device = torch.device("cuda")
    main()