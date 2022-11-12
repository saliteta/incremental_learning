import torch.nn as nn
from utils import load_models
from config import *
from train import train

from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

# construct the full dataset
dataset = ImageFolder("image-folders",...)
# select the indices of all other folders
idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] not in dataset.class_to_idx['class_s']]
# build the appropriate subset
subset = Subset(dataset, idx)




if __name__ == '__main__':

    backbone, classifier = load_models(LOAD_MODEL_PATH)
    loss = nn.CrossEntropyLoss()
    train(loss, backbone, classifier)




