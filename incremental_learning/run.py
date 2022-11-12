import torch.nn as nn
from utils import load_models
from config import *
from train import train






if __name__ == '__main__':
    subset = [ 'conference_room',  'classroom', 'bedroom', 'bathroom']
    subset_idx = [i for i in range(len(subset))]
    backbone, _ = load_models(LOAD_MODEL_PATH)
    classifier = nn.Linear(2048, len(subset)).cuda()
    loss = nn.CrossEntropyLoss()
    train(loss, backbone, classifier, subset,subset_index=subset_idx)




