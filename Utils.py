import os 
import random 
import cv2
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
from sklearn.utils import shuffle
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense,Conv2D,Flatten  
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader



#### pytorch program ####
"""
深度学习图片卷积输出大小计算公式

先定义几个参数

输入图片大小 W×W
Filter大小 F×F
步长 S
padding的像素数 P
于是我们可以得出

N = (W − F + 2P )/S+1

输出图片大小为 N×N

"""
# model build
class NVIDIA_NetworkDense(nn.Module):

    def __init__(self):
        super(NVIDIA_NetworkDense, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            # nn.Dropout(0.25)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64*1*18, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=1)
        )
        
    def forward(self, input):  
        input = input.view(input.size(0), 3, 66, 200)
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output


class NetworkLight(nn.Module):

    def __init__(self):
        super(NetworkLight, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 48, 3, stride=2),
            nn.MaxPool2d(4, stride=4),
            nn.Dropout(p=0.25)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=48*4*19, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )
        

    def forward(self, input):
        input = input.view(input.size(0), 3, 70, 320)
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output






#### end ####




"""

##### Name #####
def getName(filepath):
    return filepath.split('\\')[-1]



##### Data Importation #####
def importData(path):
    columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path,'driving_log.csv'),names=columns)
    data['Center'] = data['Center'].apply(getName)
    print('Total Images Imported:', data.shape[0])
    return data

"""

##### Balance the Data ##### 
def balancedData(data,display=True):
    nBins = 31
    samplesPerBin = 1000
    print(data[1])
    hist, bins = np.histogram(data[1],nBins)
    
    if display:
        center = (bins[:-1] + bins[1:])*0.5
        plt.bar(center, hist, width=0.06)
        plt.plot((-1,1),(samplesPerBin,samplesPerBin))
        plt.show()
    
    removeIndexList = []
    for j in range(nBins):
        binDataList = []
        for i in range(len(data(1))):
            if data[1][i] >= bins[j] and data[1][i] <= bins[j+1]:
                binDataList.append(i)
        binDataList  = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeIndexList.extend(binDataList)
    print('Removed Images: ', len(removeIndexList))
    data.drop(data.index[removeIndexList], inplace=True)
    print('Remaining Images: ', len(data))
    
    if display:
        hist, _ = np.histogram(data['Steering'], nBins)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1,1),(samplesPerBin, samplesPerBin))
        plt.show()
        
    return data


"""
##### Load the Data #####
def loadData(path, data):
    imagesPath = []
    steering = []
    
    for i in range (len(data)):
        indexedData = data.iloc[i]
        imagesPath.append(os.path.join(path, 'IMG', indexedData[0]))
        steering.append(float(indexedData[3]))
    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)
    return imagesPath, steering


"""


"""

##### Image Augmentation #####
def augmentImage(imgPath, steering):
    img = mpimg.imread(imgPath)

   
    ### PAN
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={'x':(-0.1, 0.1), 'y':(-0.1, 0.1)}) 
        img = pan.augment_image(img)
        
    ### ZOOM
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    
    ### BRIGHTNESS
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.3,1.2))
        img = brightness.augment_image(img)
    
    ### FLIP
    if np.random.rand() < 0.5:
        img = cv2.flip(img,1)
        steering = -steering
    
    
    return img, steering



##### Preprocessing the Image #####
def preProcessing(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.resize(img,(200,66))
    img = img /255
    return img

# imgRe = preProcessing(mpimg.imread('test.jpg'))
# plt.imshow(imgRe)
# plt.show()



##### Generating Batch #####
def batchGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []
        
        for i in range(batchSize):
            index = random.randint(0, len(imagesPath)-1)
            if trainFlag:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            #img , steering = augmentImage(imagesPath[index], steeringList[index])
            img = preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield(np.asarray(imgBatch),np.asarray(steeringBatch))
 

 
##### Model Creation #####        
def createModel():
    model = Sequential()
    
    model.add(Conv2D(24,(5,5),(2,2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Conv2D(36,(5,5),(2,2), activation='elu'))
    model.add(Conv2D(48,(5,5),(2,2), activation='elu'))
    model.add(Conv2D(64,(3,3), activation='elu'))
    model.add(Conv2D(64,(3,3), activation='elu'))
    
    model.add(Flatten())
    model.add(Dense(100,activation='elu'))
    model.add(Dense(50,activation='elu'))
    model.add(Dense(10,activation='elu'))
    model.add(Dense(1))
    
    model.compile(Adam(learning_rate=0.0001),loss='MSE')
    
    return model
    
    """