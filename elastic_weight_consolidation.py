'''
    EWC: Elastic weight consolidation, it is also a method used to do the incremental learning
    The basic idea is that we first find which parameter is important to the prediction of the 
    original model.
    In principle, we do not want to change those parameters. However, we can change some other 
    parameters that has nothing to do with the original model.

    To measure the how important a parameter is to original model, we need to use a mathmatical
    method which called Fisher Information Matrix.

    The details of Fisher Information Matrix is still unknow for me right now. Therefore, I will
    directly copy the code of Fisher Information Matrix from an implementation of EWC
    
    new Loss = original_loss + SIGMA(Fisher Matirx)(how much it deviate from the original model(L2))
'''
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from config import *
from tqdm import trange
from test import eval_model
from display import get_normalize_cm
from loss_function import build_scheduler, ewc_loss
from utils import load_data, load_checkpoint, load_original_classes, increase_classfier, combined_model




def ewc_train(loss_function, model, subset = None, subset_index = None, backbone_store_path = BACKBONE_STORE_PATH, 
classifier_store_path = CLASSIFIER_STORE_PATH, eval_subset = None, eval_subset_index = None):
    train_data = load_data(DATASET+'/val', BATCHSIZE, WORKERS_NUMBER, subset=subset, subset_idx=subset_index)
    params = list(model.parameters()) 
    optimizer = optim.SGD(params, LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    LR = LR_Params()
    learning_record = []
    lr_scheduler = build_scheduler(LR, optimizer, len(train_data))

    best_acc = 0
    for epoch in trange(EPOCHS):
        model.train()
        print(f'epochs {epoch}: ')
        '''
            Display of the model info
        '''


        average_loss = 0
        count = 0

        #if epoch < 10:
        #    for param in backbone.parameters(): param.requires_grad = False
        #else:
        #    for param in backbone.parameters(): param.requires_grad = True
        

        for i, (input, target) in (enumerate(train_data)):
            count += 1


            target = target.cuda()
            img = torch.autograd.Variable(input).cuda()
            logit = model(img)

            loss = loss_function(logit, target, model)
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch*len(train_data)+i))
            loss.backward()
            optimizer.step()

            average_loss += loss.detach().cpu()

        average_loss /= count
        learning_record.append(average_loss)
        print(f'the loss is {average_loss}')
        model.eval()


        
        backbone = nn.Sequential(*list(model.children())[:-1])
        classifier = nn.Sequential(*list(model.children())[-1:])
        eval_target, eval_logit, acc = eval_model(backbone, classifier, eval_subset, subset_idx=eval_subset_index)
        cm = get_normalize_cm(eval_target, eval_logit)
        np.save(f'ewc_result/cm/{epoch}.npy', cm)
        if acc>best_acc:
            best_acc = acc
            torch.save(backbone.state_dict(), backbone_store_path)
            torch.save(classifier.state_dict(), classifier_store_path)
        print(f'best_acc update at epochs {epoch}: {best_acc}')
        
    return 

if __name__ == '__main__':

    original_clases = load_original_classes('model/4')
    backbone, classifier = load_checkpoint('model/4', 4)
    classifier = increase_classfier(classifier, 2) # increase the classifier from 4

    new_classes = ['bedroom', 'lab']

    model = combined_model(backbone, classifier)


    subset_classes = new_classes # corresponding new classes list
    subset_classes_idx = [4,5]   # set correspoding new classes index
    original_data = load_data(DATASET+'/val', BATCHSIZE, WORKERS_NUMBER, subset=['bathroom','kitchen', 'furniture_store', 'office', 'bedroom', 'lab'], subset_idx=[0,1,2,3,4,5])
    loss_function = ewc_loss(model, original_data,weight=2000)
    ewc_train(loss_function, model, subset=subset_classes, subset_index=subset_classes_idx, backbone_store_path='ewc_result/backbone.tar', 
    classifier_store_path='ewc_result/classifier.tar', eval_subset=['bathroom','kitchen', 'furniture_store', 'office', 'bedroom', 'lab'],
    eval_subset_index=[0,1,2,3,4,5])    