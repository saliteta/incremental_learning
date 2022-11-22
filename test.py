import torch.nn.functional as F
import torchvision.transforms as transforms
import PIL.Image as Image
from utils import calculate_logit, load_data, load_models
from config import *
import numpy as np
from tqdm import tqdm




def test_model(backbone, classifier, img):
    '''
        The test model only give one img in and out
        What if want a batch of image?

        return vaule:
        probs and idx
        sorted by the probs value
        idx is the correspomding idx of the sorted probability
        both of them are in cuda
    '''
    backbone.eval()
    classifier.eval()
    centre_crop = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    img = Image.open(img).convert('RGB')
    img = centre_crop(img).unsqueeze(0).cuda()
    logit = calculate_logit(backbone, classifier, img)

    class_vector = F.softmax(logit, 1).data.squeeze()
    probs, idx = class_vector.sort(0, True)
    print(probs.is_cuda)
    return probs, idx

def eval_model(backbone, classifier, subset = None, subset_idx = None, save_dict_path = None):
    '''
        If you want to save the index to class name dict
        then specify the path name in save_dict_path
    '''

    '''
        the all logit will return a size of dataset_size * classsize
        the target will return 0 - (classize-1) 
        in sequence(alphabetically)
    '''
    backbone.eval()
    classifier.eval()

    test_data = load_data(DATASET+'/train', EVAL_BACHSIZE,WORKERS_NUMBER,False,False, subset = subset, subset_idx= subset_idx, save_idx_to_class= save_dict_path)

    all_target = []
    all_logit = []
    count = 0
    for i, (input, target) in (enumerate(tqdm(test_data))):
        target = target
        img = input.cuda()
        #img = torch.autograd.Variable(input).cuda()
        logit = calculate_logit(backbone, classifier, img)
        if (len(all_logit)) == 0:
            all_logit = logit.detach().cpu().numpy()
            all_target = target.detach().cpu().numpy()
        else:
            all_logit = np.concatenate((all_logit, logit.detach().cpu().numpy()), axis=0)
            all_target = np.concatenate((all_target, target.detach().cpu().numpy()), axis=0)
        count += BATCHSIZE
    
    acc = np.sum(np.argmax(all_logit, axis=1) == all_target)/len(all_target)
    print(f'acc: {acc}')
    return all_target, all_logit, acc




if __name__ == '__main__':
    backbone, classifier = load_models('model\\resnet50_0obj_19cls__SUN_RGBD_best.pth.tar')
    target, logit, acc = eval_model(backbone, classifier)
    np.save('./numpy_analysis/target.npy', target)
    np.save('./numpy_analysis/logit.npy', logit)
