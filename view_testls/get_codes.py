import matplotlib.pyplot as plt

import  torch.utils.data as data_utils
import torch
import torch.nn as nn
#from torchvision.models import inception
#from torchvision.models import Inception3
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from torchvision import datasets,models,transforms
import torch.optim as optim

from nndev import model_pip
import numpy as np
import random
import gc

import copy
import torch
import numpy as np
from torch.autograd import Variable
import sys
import os
import scipy
import cv2
from PIL import Image

res = models.resnet18(pretrained=True)
res.fc = nn.Linear(res.fc.in_features,8)

# for i,j in res.named_parameters():
# 	print(i)
# 	print(j.size())



obj1 = model_pip(model_in=res,data_path = '/data/gabriel/VC_16/SET10/dataset/',
                         batch_size=128, gpu=0,scale = True,use_gpu=True,resume=False,verbose=False)

obj1.load_model('/data/gabriel/VC_16/SET10/ResNetModel_pretrained_8_views_bs_100_e_50_10082017_0343_0.001.pth.tar')



import pickle

a = torch.zeros(10,10)
#print(obj1(Variable(a)))
#obj1.load_model(p2+'/'+m_name)
dsets,dset_loaders,dset_sizes = obj1.transform(rand=False,test_only=True)


code_dic={i:[] for i in range(0,8)}

#code_dic={}
obj1.model=obj1.model.cuda()
counter=0

label_code = np.zeros((dset_sizes['test'],9))

for data in dset_loaders['test']:
	
	img,label = data

	img = Variable(img.cuda())

	output = obj1.model(img)
	#print(label)
	#print(output.cpu().data.numpy())
	#print(output.cpu().data[0].view(1,-1).numpy())
	label_code[counter*128:128*(counter+1),1:] = output.cpu().data.numpy()
	label_code[counter*128:128*(counter+1),0] = label.numpy()


	counter+=1

import pandas as pd
np.save('./bla.npy',label_code)
df = pd.DataFrame(label_code)

df.to_csv('/data/gabriel/tsne/train1.csv',header=['label','one','two','three','four','five','six','seven','eight'])
#np.savetxt("/data/gabriel/tsne/train1.csv",label_code,delimiter=',')


