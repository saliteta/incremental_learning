import os
import numpy as np
import torch.nn as nn
from test import eval_model
from utils import load_models, load_checkpoint
from config import *
from train import train
from display import base_class_selection




if __name__ == '__main__':
    class_list = base_class_selection()
    for i in range(10):
        subset = []
        for j in range(4+i):
            subset.append(class_list[j][1])
        subset_idx = [i for i in range(len(subset))]
        backbone, _ = load_models(LOAD_MODEL_PATH)
        classifier = nn.Linear(2048, len(subset)).cuda()
        loss = nn.CrossEntropyLoss()
        os.mkdir(f'./model/{len(subset)}')
        with open(f'./model/{len(subset)}/description.txt', 'w') as f:
            for i in subset:
                f.writelines(i+'\n')
        train(loss, backbone, classifier, subset,subset_index=subset_idx, backbone_store_path=f'./model/{len(subset)}/backbone.tar',classifier_store_path=f'./model/{len(subset)}/classifier.tar')
        backbone, classifier = load_checkpoint(f'./model/{len(subset)}', linear_node_size=len(subset))
        target, logit, acc = eval_model(backbone, classifier, subset=subset, subset_idx=subset_idx)
        np.save(f'./model/{len(subset)}/target.npy', target)
        np.save(f'./model/{len(subset)}/logit.npy', logit)



