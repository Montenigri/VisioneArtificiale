import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
import glob
import random 
random.seed(42)
np.random.seed(42) 


def load_groups(input_folder): 
    ''' 
    Loading the list of sub-folders into a python list with their 
    corresponding label. 
     
    '''
    listDir = os.listdir(input_folder)
    listDir.sort()
    groups = []
    index = 0
    for dir in listDir:
        videos =  glob.glob(f"{input_folder}/{dir}/*/*.mpg")
        groups.append([videos,index]) 
        index+=1
    return groups


def get_file_list(train_groups): 
    train = [] 
    for video_dir in train_groups: 
        if os.path.isfile(video_dir[0]): 
            ext = os.path.splitext(video_dir[0])[1] 
            if (ext == '.mpg'): 
                video_reader = cv2.VideoCapture(video_dir[0]) 
                if video_reader and  video_reader.get(cv2.CAP_PROP_FRAME_COUNT) >= 16: 
                    train.append([video_dir[0], video_dir[1]]) 
    return train

def split_data(groups): 
    ''' 
    Splitting the data at random for train and test set. 
    ''' 
    group_count = len(groups) 
    indices = np.arange(group_count) 
 
    np.random.shuffle(indices) 
 
    # 60% training, 20% val and 20% test. 
    train_count = int(0.6 * group_count) 
    val_ind = int(0.8 * group_count) 
 
    train_groups = [groups[i] for i in indices[:train_count]] 
    val_groups  = [groups[i] for i in indices[train_count:val_ind]] 
    test_groups  = [groups[i] for i in indices[val_ind:]] 
 
    train = get_file_list(train_groups) 
    val = get_file_list(val_groups) 
    test = get_file_list(test_groups) 
 
    return train, val, test


#Manca la label
#Far ritornare liste di dimensione della batch desiderata
#In pratica aggiungere un parametro dimensione in input
#Fare shuffle della lista

def gen(lista):
    while True:
        results = []
        for i in len(lista):
            result = []
            src = cv2.VideoCapture(i[0])
            ret, frame = src.read()
            count = 0
            while ret:
                result.append(cv2.resize(frame, (160, 120)))
                count+=1
            results.append(random.sample(result, 30))
        yield results


groups = load_groups('UCF11_updated_mpg') 
train, val, test = split_data(groups) 