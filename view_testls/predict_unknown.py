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
# import pickle

# a = torch.zeros(10,10)
# #print(obj1(Variable(a)))
# #obj1.load_model(p2+'/'+m_name)
# dsets,dset_loaders,dset_sizes = obj1.transform(rand=False,test_only=True)

# import os

# U_list = (i for i in os.listdir('/data/gabriel/U_and_Off/U/') if '.jpg' in i)
# off_list = (i for i in os.listdir('/data/gabriel/U_and_Off/Off/') if '.jpg' in i)

# data_transforms = transforms.Compose([transforms.Scale(300),
# 					transforms.RandomCrop(300),

# 					transforms.ToTensor(),
# 					transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) ])

# tensor_list_u = []
# tensor_list_off = []

# from PIL import Image
# #print(data_transforms)

# dic_u={}
# dic_off={}

# for i in U_list:
# 	img = Image.open('/data/gabriel/U_and_Off/U/'+i)
# 	tensor_list_u.append((Variable(data_transforms(img).view(1,3,300,300)),i[:i.find('.jpg')]) )

# 	#print(tensor_list)
# 	#break

# for i in off_list:
# 	img = Image.open('/data/gabriel/U_and_Off/Off/'+i)
# 	tensor_list_off.append((Variable(data_transforms(img).view(1,3,300,300)),i[:i.find('.jpg')]) )
	
# 	#print(tensor_list)
# 	#break


# torch.save(tensor_list_u,'./tensor_list_u.pth')
# torch.save(tensor_list_off,'./tensor_list_off.pth')


obj1.model=obj1.model.cuda()
counter=0

label_code = []
label = []


tensor_list_u = torch.load('./tensor_list_u.pth')
tensor_list_off = torch.load('./tensor_list_off.pth')

codes_u = []
codes_off = []

for img,name in tensor_list_u:
	img = img.cuda()
	output = obj1.model(img)
	print(output.cpu().data)
	break
	codes_u.append(output.cpu().data)
	print(output)
	print(name)

'''
for img,name in tensor_list_off:
	img = img.cuda()
	output = obj1.model(img)

	codes_off.append(output.cpu().data)
	print(output.cpu().data)

	print(output)
	print(name)

'''
# for data in dset_loaders['test']:
	
# 	img,label = data

# 	img = Variable(img.cuda())

# 	output = obj1.model(img)
# 	#print(label)
# 	#print(output.cpu().data.numpy())
# 	#print(output.cpu().data[0].view(1,-1).numpy())
# 	label_code[counter*128:128*(counter+1),1:] = output.cpu().data.numpy()
# 	label_code[counter*128:128*(counter+1),0] = label.numpy()


# 	counter+=1

# import pandas as pd
# np.save('./bla.npy',label_code)
# df = pd.DataFrame(label_code)

# df.to_csv('/data/gabriel/tsne/train1.csv',header=['label','one','two','three','four','five','six','seven','eight'])
#np.savetxt("/data/gabriel/tsne/train1.csv",label_code,delimiter=',')


