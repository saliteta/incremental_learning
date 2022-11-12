import torch
import torch.optim as optim
import torch.nn as nn
from test import eval_model
from utils import load_data, calculate_logit, load_models
from config import *
from loss_function import build_scheduler


def train(loss_function, backbone, classifier, subset = None, subset_index = None):
    train_data = load_data(DATASET+'/train', BATCHSIZE, WORKERS_NUMBER, subset=subset, subset_idx=subset_index)
    params = list(backbone.parameters()) + list(classifier.parameters()) 
    optimizer = optim.SGD(params, LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    LR = LR_Params()
    learning_record = []
    lr_scheduler = build_scheduler(LR, optimizer, len(train_data))


    for epoch in range(EPOCHS):
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
        best_acc = 0

        

        for i, (input, target) in (enumerate(train_data)):
            count += 1
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch*len(train_data)+i))

            target = target.cuda()
            img = torch.autograd.Variable(input).cuda()
            logit = calculate_logit(backbone, classifier, img)
            probability = nn.functional.softmax(logit, dim = -1)

            loss = loss_function(probability, target)

            loss.backward()
            optimizer.step()

            average_loss += loss.detach().cpu()

        average_loss /= count
        learning_record.append(average_loss)
        print(f'the loss is {average_loss}')
        
        _, _, acc = eval_model(backbone, classifier, subset)
        if acc>best_acc:
            best_acc = acc
            torch.save(backbone.state_dict(), BACKBONE_STORE_PATH)
            torch.save(classifier.state_dict(), CLASSIFIER_STORE_PATH)
        print(f'best_acc update at epochs {epoch}: {best_acc}')
        

    return 

if __name__ == '__main__':
    backbone, classifier = load_models(LOAD_MODEL_PATH)
    loss = nn.CrossEntropyLoss()
    train(loss, backbone, classifier)




