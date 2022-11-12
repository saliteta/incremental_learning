import os
import time
import torch
import torch.nn as nn
import PIL.Image as Image
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from config import IMG_EXTENSIONS
from torchvision.datasets import DatasetFolder

def increase_classfier(linear_layer, node_number = 1):
    '''
    Linear layer is the classifier layer
    node number is how many new class you want to increase, default value is 1

    This function return you a classfier layer with origin weight plus a new weight for new node
    '''
    if next(linear_layer.parameters()).is_cuda:
        linear_layer = linear_layer.cpu()

    origin_weight = linear_layer.weight
    origin_bias = linear_layer.bias
    origin_shape = linear_layer.weight.shape

    new_layer = nn.Linear(origin_shape[1], origin_shape[0] + node_number)
    new_layer.weight = nn.parameter.Parameter(torch.cat((origin_weight, new_layer.weight[0:node_number])))
    new_layer.bias = nn.parameter.Parameter(torch.cat((origin_bias, new_layer.bias[0:node_number])))
    
    return new_layer.cuda()

def load_checkpoint(checkpoint, base_model, obj_model, classifier):
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    if classifier:
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
    if base_model:
        base_model.load_state_dict(checkpoint['model_state_dict'])
    if obj_model:
        obj_model.load_state_dict(checkpoint['obj_state_dict'])
    return base_model, obj_model, classifier, start_epoch, best_prec1


def load_models(checkpoint_path = 'model\SUN_RGBD_RESNET.pth.tar'):
    '''this model take a str as an input
        it return the backbone of a resnet50 and corresponding classifier
        if load the places 365 model, no pretrained classifier
    '''
    state_dict = torch.load(checkpoint_path)
    model = models.resnet50(pretrained = None)
    model = nn.DataParallel(model).cuda()
    classifier = nn.Linear(2048,19)
    try:
        model.load_state_dict(state_dict['model_state_dict'])
        state_dict['classifier_state_dict']['weight'] = state_dict['classifier_state_dict'].pop('fc.weight')
        state_dict['classifier_state_dict']['bias'] = state_dict['classifier_state_dict'].pop('fc.bias')
        classifier.load_state_dict(state_dict['classifier_state_dict'])
        classifier = classifier.cuda()
    except:
        model = models.resnet50(pretrained = None, num_classes = 365)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in state_dict['state_dict'].items()}
        model.load_state_dict(state_dict)
        model = nn.Sequential(*list(model.children())[:-1])
        model = nn.DataParallel(model).cuda()
        classifier = classifier.cuda()




    return model, classifier

def calculate_logit(backbone, classifier, img):
    backbone_features = backbone(img)
    backbone_features = backbone_features.view(backbone_features.size(0), -1) #transform the size to (batchsize, 2048)
    logits = classifier(backbone_features)
    return logits


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class specified_class_dataset(DatasetFolder):

    def __init__(self, root, loader = pil_loader, extensions=None, transform=None, target_transform=None, is_valid_file=None, class_names= None,class_idx = None):

        self.class_names = class_names
        self.class_idx = class_idx
        super().__init__(root, loader = pil_loader, extensions =IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)



    def _find_classes(self, dir):
        if self.class_names!=None:
            class_to_idx = {}
            for i in range(len(self.class_idx)):
                class_to_idx[self.class_names[i]] = self.class_idx[i]
            return self.class_names, class_to_idx


        return super()._find_classes(dir)





def load_data(data_dir, batchsize, workers, train=True, shuffle=True, subset = None, subset_idx = None):

    '''
        If we add some subset, and do the incremental learning, we want our subset_idx happend to be the last few idx
        For example, previously, we have already train the classes a,b,c whose corresponding index is 0, 1, 2. 
        Now, we want to add more classes like alpha and beta, than the index should be 4, and 5
        This implementation is becuause we can only add linear layer as the last node
    '''


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if train:
        dataset = specified_class_dataset(data_dir, transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            normalize,
        ]),class_names=subset, class_idx=subset_idx)
    else: # did we use this method during test?
        dataset = specified_class_dataset(data_dir, transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), class_names=subset, class_idx=subset_idx)


    data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size = batchsize, shuffle=shuffle,
            num_workers = workers, pin_memory=True)

    return data_loader

def get_classes_name_list(parant_folder_path):
    directory_list = list()
    for _, dirs, _ in os.walk(parant_folder_path, topdown=False):
        for name in dirs:
            directory_list.append(name)
    return directory_list

def logging(original_class_list, name_new_classes, original_classes_folder, new_classes_folder):
    with open('log.txt', 'a') as f:
        f.writelines(original_class_list, original_classes_folder, time.ctime)
        f.writelines(name_new_classes,new_classes_folder)




if __name__ == '__main__':
    dataloader = load_data("/data/dataset/SUN_RGBD/train",16,2,subset=['computer_room',  'corridor', 'bedroom'], subset_idx=[0,1,2])
    for (input, label) in dataloader:
        print(len(label))
        print(type(label))