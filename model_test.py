import numpy as np
import torch.nn as nn
from config import *
from train import train
from display import draw_confusion_matrix
from test import eval_model
from utils import load_checkpoint, load_original_classes, increase_classfier, combined_model


'''
    what we are going to do is training a model with more than its original class,
'''


if __name__ == '__main__':


    backbone, classifier = load_checkpoint('lwf_result', 6)
    model = combined_model(backbone, classifier)
    classifier = nn.Sequential(*list(model.children())[-1:])
    
    for name, parameters in classifier.named_parameters():
        print(name)