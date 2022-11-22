import torch
import torch.optim as optim
import torch.nn as nn
import os
from test import eval_model
from utils import load_data, calculate_logit, load_models
from config import *
from loss_function import build_scheduler
from tqdm import trange


def train(loss_function, backbone, classifier, subset = None, subset_index = None, backbone_store_path = BACKBONE_STORE_PATH, classifier_store_path = CLASSIFIER_STORE_PATH):
    train_data = load_data(DATASET+'/val', BATCHSIZE, WORKERS_NUMBER, subset=subset, subset_idx=subset_index)
    params = list(backbone.parameters()) + list(classifier.parameters()) 
    optimizer = optim.SGD(params, LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    LR = LR_Params()
    learning_record = []
    lr_scheduler = build_scheduler(LR, optimizer, len(train_data))

    best_acc = 0
    for epoch in trange(EPOCHS):
        print(f'epochs {epoch}: ')
        '''
            Display of the model info
        '''

        if BACKBONE_TRAIN:
            backbone.train()
        else:
            backbone.eval()
        if CLASSIFIER_TRAIN:
            classifier.train()
        else:
            classifier.eval()
        # set wether train the backbone or classifier

        average_loss = 0
        count = 0

        if epoch < 10:
            for param in backbone.parameters(): param.requires_grad = False
        else:
            for param in backbone.parameters(): param.requires_grad = True
        

        for i, (input, target) in (enumerate(train_data)):
            count += 1


            target = target.cuda()
            img = torch.autograd.Variable(input).cuda()
            logit = calculate_logit(backbone, classifier, img)

            loss = loss_function(logit, target)
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch*len(train_data)+i))
            loss.backward()
            optimizer.step()

            average_loss += loss.detach().cpu()

        average_loss /= count
        learning_record.append(average_loss)
        print(f'the loss is {average_loss}')
        with open('log/log.txt', 'a') as f:
            content = f'epoch number: {epoch} | batch number: {i} \n learning rate:' + str(optimizer.param_groups[0]['lr'])+'\n'
            f.write(content)
        _, _, acc = eval_model(backbone, classifier, subset, subset_idx=subset_index)
        if acc>best_acc:
            best_acc = acc
            torch.save(backbone.state_dict(), backbone_store_path)
            torch.save(classifier.state_dict(), classifier_store_path)
        print(f'best_acc update at epochs {epoch}: {best_acc}')
        

    return 

if __name__ == '__main__':
    backbone, classifier = load_models(LOAD_MODEL_PATH)
    loss = nn.CrossEntropyLoss()
    train(loss, backbone, classifier)




