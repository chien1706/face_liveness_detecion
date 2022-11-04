from torch.utils.data import DataLoader, Dataset
import glob
import pandas as pd
import cv2
import torch
import imgaug.augmenters as iaa
import random
color_augment_instance = iaa.OneOf([
        iaa.AddToHueAndSaturation((-20, 20)),
        iaa.Grayscale(alpha=(0.0, 0.5)),
        iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
        iaa.AddToHueAndSaturation((-20, 20), per_channel=True),
        iaa.RemoveSaturation((0.0, 0.5))
    ])

contrast_augment_instance = iaa.OneOf([
    iaa.GammaContrast((0.5, 2.0)),
    iaa.SigmoidContrast(gain=(3, 5), cutoff=(0.2, 0.4)),
    iaa.LinearContrast((0.4, 1.6))
])
blur_augment_instance = iaa.OneOf([
    iaa.GaussianBlur(sigma=1.5),
    iaa.AverageBlur(k=3),
    iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)),

])
arithmetic_instance = iaa.OneOf([
    iaa.AddElementwise((-20, 20)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
    iaa.AdditiveLaplaceNoise(scale=(0, 0.05 * 255)),
    iaa.AdditivePoissonNoise(lam=(0, 10)),
    iaa.Multiply((0.7, 1.2)),
    iaa.MultiplyElementwise((0.7, 1.2)),
    iaa.ImpulseNoise(0.05),
    iaa.SaltAndPepper(0.05),
    iaa.Add((-10, 10), per_channel=0.5),  # change brightness of images (by -10 to 10 of original value)
])
augmenters = [color_augment_instance, contrast_augment_instance, blur_augment_instance, arithmetic_instance]
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
class FLDDataset(Dataset):
    def __init__(self, path, img_size = 288):
        self.path = path
        self.img_size = img_size
        self.video_paths = glob.glob(path + "/videos/*.mp4")
        # print(self.video_paths)
        self.label_path = path + "/label.csv"
        self.labels = pd.read_csv(self.label_path)
        self.labels.set_index("fname", inplace=True)
    def __len__(self):
        return len(self.video_paths)
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split("/")[-1]

        label = int(self.labels.loc[video_name])
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        _, frame = cap.read()
        # print(frame.shape)
        # frame [h, w, c]
        # print(frame)
        frame = resizeKeepRatio(frame, self.img_size)
        # auger = random.choice(augmenters)
        # frame = auger(image = frame)
        cv2.imwrite("debug.png", frame)
        frame = frame / 255.0
        frame = torch.tensor(frame, dtype = torch.float32).permute(2,0,1)
        return torch.tensor(label, dtype = torch.float32), frame
    
if __name__ == '__main__':
    dataset = FLDDataset("/home/aimenext/luantt/zaloai/face_liveness_detection/dataset/train")
    label, image = dataset[1]
    print(image, label)
        