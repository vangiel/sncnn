import os
import sys

import torch
import pickle
import cv2
import argparse
from torch.utils.data import random_split

from dataloader import ImageDataset


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--resolution", type=int, required=False, default=360,
                help="Resolution of the input image, defaoult 360 (360x360)")
ap.add_argument("-ts", "--train_split", type=float, required=False, default=0.9,
                help="Percentage of train split, default 0.9")
ap.add_argument("-m", "--mode", type=str, required=False, default="train",
                help="Mode train or test")
args = vars(ap.parse_args())

# Check if the dataset is already loaded:
path_save = "./datasets/video2map"
try:
    os.mkdir(path_save)
except OSError as error:
    print("The dataset has been already loaded, please delete video2map if you want to reload")
    sys.exit(0)

# define the train and test splits
TRAIN_SPLIT = args['train_split']
TEST_SPLIT = (1 - TRAIN_SPLIT)

img_size = (args['resolution'], args['resolution'])

data = ImageDataset("images_dataset", img_size, args['mode'], transform=None, target_transform=None, debug=False)

numTrainSamples = int(len(data) * TRAIN_SPLIT)
numTestSamples = len(data) - numTrainSamples

(trainData, testData) = random_split(data,
                                     [numTrainSamples, numTestSamples],
                                     generator=torch.Generator().manual_seed(42))

if __name__ == "__main__":
    counter = 0
    p_trainA = os.path.join(path_save, "trainA")
    os.mkdir(p_trainA)
    p_trainB = os.path.join(path_save, "trainB")
    os.mkdir(p_trainB)
    for img, label in trainData:
        filename = "train_" + str(counter).zfill(6) + ".jpg"
        os.chdir(p_trainA)
        with open(filename, "wb") as f_out:
            pickle.dump(img, f_out)
        os.chdir("../../../")
        os.chdir(p_trainB)
        cv2.imwrite(filename, label)
        os.chdir("../../../")
        counter += 1

    counter = 0
    p_testA = os.path.join(path_save, "testA")
    os.mkdir(p_testA)
    p_testB = os.path.join(path_save, "testB")
    os.mkdir(p_testB)
    for img, label in testData:
        filename = "test_" + str(counter).zfill(6) + ".jpg"
        os.chdir(p_testA)
        with open(filename, "wb") as f_out:
            pickle.dump(img, f_out)
        os.chdir("../../../")
        os.chdir(p_testB)
        cv2.imwrite(filename, label)
        os.chdir("../../../")
        counter += 1
