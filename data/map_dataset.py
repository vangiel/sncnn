import sys
import os

import numpy as np
import torch
from data.base_dataset import BaseDataset
import cv2
from pathlib import Path

import h5py


LIMIT = 200000


class MapDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.img_labels = []
        self.images = []
        self.names = []
        self.img_dir = opt.dataroot
        self.image_size = (opt.crop_size, opt.crop_size)
        self.path_save = "cache"
        self.mode = opt.phase
        if self.mode == "test":
            self.limit = 20
        else:
            self.limit = LIMIT

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

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

            self.names.append(file)

            label_image = cv2.imread(str(file), 0)
            label_image = cv2.resize(label_image, self.image_size, interpolation=cv2.INTER_AREA)
            label_image = torch.from_numpy(label_image).type(torch.FloatTensor)
            label_image = (label_image/255.)*2.-1.
            self.img_labels.append(label_image[None, :, :])

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

            for i in range(len(final_images)):
                final_images[i] = cv2.resize(final_images[i], self.image_size, interpolation=cv2.INTER_AREA)

            image = np.concatenate(final_images, axis=2)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).type(torch.FloatTensor)
            image = (image/255.)*2.-1.
            self.images.append(image)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        AB_path = os.path.join(self.img_dir, self.names[idx])
        image = self.images[idx]
        label = self.img_labels[idx]
        return {'A': image, 'B': label, 'A_paths': AB_path, 'B_paths': AB_path}

    def get_dataset_name(self):
        i, j = self.image_size
        dataset_name = "dataset_images" + '_' + self.mode + '_size_' + str(i) + "x" + str(j) + '_s_' + str(self.limit) \
                       + '.hdf5'
        return dataset_name

    def save(self):
        if self.mode == "test":
            return

        # Generate paths
        dataset_path = self.path_save + "/" + self.get_dataset_name()
        print("Saving dataset into: ", dataset_path)
        os.makedirs(self.path_save, exist_ok=True)
        images = []
        labels = []
        for i in range(len(self.images)):
            images.append(self.images[i].detach().to('cpu').numpy())
            labels.append(self.img_labels[i].detach().to('cpu').numpy())

        # Save images and labels
        with h5py.File(dataset_path, 'w') as file:
            file.create_dataset('images', data=np.array(images))
            file.create_dataset('labels', data=np.array(labels))

    def load(self):
        # Generate paths
        dataset_path = self.path_save + "/" + self.get_dataset_name()
        print("Loading dataset from cache...")

        # Load dataset
        with h5py.File(dataset_path, 'r') as file:
            images = file['images']
            labels = file['labels']
            for i in range(len(images)):
                image = torch.from_numpy(images[i]).type(torch.FloatTensor)
                label = torch.from_numpy(labels[i]).type(torch.FloatTensor)
                self.images.append(image)
                self.img_labels.append(label)

    def has_cache(self):
        # Generate paths
        dataset_path = self.path_save + "/" + self.get_dataset_name()
        if self.mode == "test":
            return False
        return os.path.exists(dataset_path)
