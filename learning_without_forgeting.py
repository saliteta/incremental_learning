'''
    LWF: learning witout forgetting, basically is a method base on the distillation
    This method add more in the loss function so that the model not only focus on the 
    current data, but also focus on preserve the result of the old training result. 

    That is: loss_origin + (prediction of old model - prediction of current model)
'''


import torch
import numpy as np
import torch.optim as optim

from config import *
from tqdm import trange
from test import eval_model
from loss_function import build_scheduler, lwf_loss
from display import get_normalize_cm
from utils import load_data, calculate_logit, load_original_classes, load_checkpoint, increase_classfier



def lwf_train(loss_function, backbone, classifier, original_backbone, original_classifier,subset = None, subset_index = None, 
backbone_store_path = BACKBONE_STORE_PATH, classifier_store_path = CLASSIFIER_STORE_PATH, eval_subset = None, eval_subset_index = None):
    
    
    train_data = load_data(DATASET+'/val', BATCHSIZE, WORKERS_NUMBER, subset=subset, subset_idx=subset_index)
    params = list(backbone.parameters()) + list(classifier.parameters()) 
    optimizer = optim.SGD(params, LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    '''old model as a teacher'''
    original_backbone.eval()
    original_classifier.eval()
    '''always eval'''


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
            teacher_logit = calculate_logit(backbone=original_backbone, classifier=original_classifier,img = img)
            loss = loss_function(logit, teacher_logit, target)
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch*len(train_data)+i))
            loss.backward()
            optimizer.step()

            average_loss += loss.detach().cpu()

        average_loss /= count
        learning_record.append(average_loss)
        print(f'the loss is {average_loss}')
        eval_target, eval_logit, acc = eval_model(backbone, classifier, eval_subset, subset_idx=eval_subset_index)
        cm = get_normalize_cm(eval_target, eval_logit)
        np.save(f'lwf_result/cm/{epoch}.npy', cm)
        if acc>best_acc:
            best_acc = acc
            torch.save(backbone.state_dict(), backbone_store_path)
            torch.save(classifier.state_dict(), classifier_store_path)
        print(f'best_acc update at epochs {epoch}: {best_acc}')
        

    return 

if __name__ == '__main__':
    original_clases = load_original_classes('model/4')
    backbone, classifier = load_checkpoint('model/4', 4)
    original_backbone, original_classifier = load_checkpoint('model/4', 4)
    '''
        load a models with only 4 classes trained
    '''
    classifier = increase_classfier(classifier, 2) # increase the classifier to 4

    new_classes = ['bedroom', 'lab']
    loss_function = lwf_loss(weight=0.5)
    subset_classes = new_classes # corresponding new classes list
    subset_classes_idx = [4,5]   # set correspoding new classes index
    lwf_train(loss_function, backbone, classifier,original_backbone=original_backbone,original_classifier=original_classifier,
    subset=subset_classes, subset_index=subset_classes_idx, backbone_store_path='lwf_result/backbone.tar', 
    classifier_store_path='lwf_result/classifier.tar', eval_subset=['bathroom','kitchen', 'furniture_store', 'office', 'bedroom', 'lab'], 
    eval_subset_index=[0,1,2,3,4,5])




