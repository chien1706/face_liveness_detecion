from torch.utils.data import DataLoader, Dataset
import glob
import pandas as pd
import cv2
import torch
import imgaug.augmenters as iaa
import random

def resizeKeepRatio(im, desired_size):
    # im = cv2.imread(im)
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    return new_im
class FLDTestDataset(Dataset):
    def __init__(self, path, img_size = 288):
        self.path = path
        self.img_size = img_size
        self.video_paths = glob.glob(path + "/videos/*.mp4")

    def __len__(self):
        return len(self.video_paths)
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split("/")[-1]

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        _, frame = cap.read()
        # print(frame.shape)
        # frame [h, w, c]
        # print(frame)
        frame = resizeKeepRatio(frame, self.img_size)
        # auger = random.choice(augmenters)
        # frame = auger(image = frame)
        frame = frame / 255.0
        frame = torch.tensor(frame, dtype = torch.float32).permute(2,0,1)
        return frame, video_name
    
if __name__ == '__main__':
    dataset = FLDTestDataset("/home/aimenext/luantt/zaloai/face_liveness_detection/dataset/train")
    label, image = dataset[1]
    print(image, label)
        