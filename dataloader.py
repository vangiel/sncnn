import sys

import torch as th
from torch.utils.data import Dataset
import cv2
from pathlib import Path


class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_labels = []
        self.images = []
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        self._load()

    def _load(self):
        for file in Path(self.img_dir).glob('*.png'):
            self.img_labels.append(cv2.imread(str(file), 0))

            video_path = self.img_dir + "/" + str(file).split("/")[1].split(".")[0].split("_")[0] + ".mp4"
            print(video_path)
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                print("Error opening video stream or file")
            frames = []
            while video.isOpened():
                ret, f = video.read()
                if ret:
                    frames.append(f)
                    cv2.imshow('Frame', f)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                else:
                    break
            video.release()
            cv2.destroyAllWindows()

            print(len(frames))
            # self.images = None

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        pass
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path)
        # label = self.img_labels.iloc[idx, 1]
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        # return image, label


data = ImageDataset("images_dataset")



