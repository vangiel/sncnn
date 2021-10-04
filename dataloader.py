import sys
import os

import numpy as np
from torch.utils.data import Dataset
import cv2
from pathlib import Path

import h5py


LIMIT = 20000


class ImageDataset(Dataset):
    def __init__(self, img_dir, net, mode="train", transform=None, target_transform=None, debug=False):
        self.img_labels = []
        self.images = []
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.net = net
        self.debug = debug
        self.path_save = "cache"
        self.mode = mode
        self.limit = LIMIT

        if self.has_cache():
            self.load()
        else:
            self._load_data()
            self.save()

    def _load_data(self):
        index = 0
        for file in Path(self.img_dir).glob('*.png'):
            if index > self.limit:
                print("Stop loading data, limit reached")
                break
            if index % 100 == 0:
                print(index)
            index += 1

            self.img_labels.append(cv2.imread(str(file), 0))

            video_path = self.img_dir + "/" + str(file).split("/")[1].split(".")[0].split("_")[0] + ".mp4"
            # print("Loading ---> ", video_path)
            video = cv2.VideoCapture(video_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            if not video.isOpened():
                print("Error opening video stream or file")
            frames = []
            timestamps = []
            while video.isOpened():
                ret, f = video.read()
                if ret:
                    frames.append(f)
                    timestamps.append(video.get(cv2.CAP_PROP_POS_MSEC))
                else:
                    break
            video.release()

            final_images = [frames[-1]]
            last_t = timestamps[-1]
            for idx, frame in enumerate(reversed(frames), 1):
                if len(final_images) >= 3:
                    break
                if last_t - timestamps[-idx] >= 1000.0:
                    final_images.append(frame)
                    last_t = timestamps[-idx]

            if self.net == "VGG16":
                dims = (214, 214)
            elif self.net == "LeNet":
                dims = (28, 28)
            else:
                print("No valid network to generate the dataset.")
                sys.exit(0)

            for i in range(len(final_images)):
                final_images[i] = cv2.resize(final_images[i], dims, interpolation=cv2.INTER_AREA)

            image = np.concatenate(final_images, axis=2)
            self.images.append(image)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def get_dataset_name(self):
        dataset_name = "dataset_images" + '_' + self.mode + '_net_' + self.net + '_s_' + str(self.limit) + '.hdf5'
        return dataset_name

    def save(self):
        if self.debug:
            return

        # Generate paths
        dataset_path = self.path_save + "/" + self.get_dataset_name()
        print(dataset_path)
        print("Saving dataset into: ", dataset_path)
        os.makedirs(self.path_save, exist_ok=True)

        # Save images and labels
        with h5py.File(dataset_path, 'w') as file:
            file.create_dataset('images', data=np.array(self.images))
            file.create_dataset('labels', data=np.array(self.img_labels))

    def load(self):
        # Generate paths
        dataset_path = self.path_save + "/" + self.get_dataset_name()
        print("Loading dataset from cache...")

        # Load dataset
        with h5py.File(dataset_path, 'r') as file:
            images = file['images']
            labels = file['labels']
            for i in range(len(images)):
                self.images.append(images[i])
                self.img_labels.append(labels[i])

    def has_cache(self):
        # Generate paths
        dataset_path = self.path_save + "/" + self.get_dataset_name()
        if self.debug:
            return False
        return os.path.exists(dataset_path)





