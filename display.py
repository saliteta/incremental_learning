import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

'''
    After having the numpy labels, we need to have a corresponding 
    correct mapping from name to index, this index is stored in a 
    csv file

    According to this csv file and output logits, we obtain the confusion matrix

    Need to add two more things:
    1. CSV file
    2. logits
'''

def get_normalize_cm(target, logit):
    predict = np.argmax(logit, axis=1)
    matrix = confusion_matrix(target, predict)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    return matrix

def read_dict_csv_to_dict(dataframe_path):
    df = pd.read_csv(dataframe_path)
    index_to_class_dict = {}
    count = 0
    for key in df:
        if count == 0:
            count+=1
            continue
        index_to_class_dict[int(df[key])] = key
    return index_to_class_dict

def draw_confusion_matrix(target, logit, dict, save_path = 'logit_test/confusion_matrix.png'):
    '''
        target is a numpy array, size: batchsize*1
        logit is a numpy array, size: batchsize*classnumber
        dict is a dictionary, len: class_number, example
        {1:'class_a', 2:'class_b', ...}
        return a normalized matrix
    '''
    predict = np.argmax(logit, axis=1)
    matrix = confusion_matrix(target, predict)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    index = []
    for i in range(len(dict)):
        index.append(dict[i])
    if save_path != None: 
        plt.figure(figsize=(20,20))
        sns.heatmap(matrix, annot=True, xticklabels=index, yticklabels=index)
        plt.savefig(save_path)
    return matrix

def base_class_selection(dict_csv_path = 'logit_test/out_logit.csv', logit_path = 'logit_test/logit.npy' , target_path = 'logit_test/target.npy'):
    idx_class = read_dict_csv_to_dict(dict_csv_path)
    logit = np.load(logit_path)
    target = np.load(target_path)
    cm = draw_confusion_matrix(target, logit, idx_class, save_path=None)
    index_class_tuple = []
    for i in range(len(cm)):
        index_class_tuple.append((cm[i][i], idx_class[i]))
    index_class_tuple.sort(reverse=True)
    return index_class_tuple

if __name__ == "__main__":
    a = base_class_selection()
    print(a)


