import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *
from copy import deepcopy
from torch.autograd import Variable
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.cosine_lr import CosineLRScheduler

class lwf_loss(nn.Module):
    '''
        the difference between original model and the difference between
        new data's result    
    '''

    def __init__(self, weight = 1):
        '''final result will be weight*(loss betweem real and predict)+(loss betweem original prediction and current prediction)'''
        super().__init__()
        self.real_loss = nn.CrossEntropyLoss()
        self.predict_loss = nn.MSELoss()
        self.weight = weight
        self.softmax = nn.Softmax(dim=1)

    def forward(self,current_prediction, original_prediction, target):

        std_shape = current_prediction.size()
        zeros = torch.zeros(std_shape[0],std_shape[1]-original_prediction.size()[1]).cuda()
        original_padding = torch.cat((original_prediction,zeros), 1)
        # make the original output padding to the second one
        predict_loss = self.predict_loss((current_prediction),(original_padding))
        real_loss = self.real_loss(current_prediction, target)
        print('real_loss:',real_loss)
        print('predict_loss',predict_loss)
        return self.weight*real_loss+predict_loss

class ewc_loss(nn.Module):
    '''
        weight * original_loss + deviation_loss
        where deviation loss is how much it deviate from the original model
        this term is judge by the class EWC
    '''

    def __init__(self, model, dataset, weight = 100):
        '''final result will be (loss betweem real and predict)+weight*(loss betweem original prediction and current prediction)'''
        super().__init__()
        self.real_loss = nn.CrossEntropyLoss()
        self.deviate_loss = EWC(model, dataset)
        self.weight = weight
        self.softmax = nn.Softmax(dim=1)

    def forward(self,current_prediction, target, model):
        deviate_loss = self.weight*self.deviate_loss.penalty(model)
        real_loss = self.real_loss(current_prediction, target)
        print('real_loss:',real_loss)
        print('predict_loss',deviate_loss)
        return real_loss+deviate_loss

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for _, (input, _) in (enumerate(self.dataset)):
            self.model.zero_grad()
            input = variable(input)
            output = self.model(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss

def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.WARMUP_EPOCHS * n_iter_per_epoch)
    decay_steps = int(config.LR_SCHEDULER_DECAY_EPOCHS * n_iter_per_epoch)

    lr_scheduler = None
    if config.LR_SCHEDULER_NAME == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=config.MIN_LR,
            warmup_lr_init=config.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=2,
            cycle_decay=0.1,
            t_in_epochs=False,
        )
    elif config.LR_SCHEDULER_NAME == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config.LR_SCHEDULER_DECAY_RATE,
            warmup_lr_init=config.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    return lr_scheduler


### using the build schedular to train

if __name__ == '__main__':
    x = torch.randn(2, 3)
    y = torch.zeros(2,2)# bacthsize the same, 2 is the new class number
    z = torch.cat((x,y),1)
    print(z.size())