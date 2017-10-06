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
from nndev import model_pip
from collections import OrderedDict
import pickle
def s_points(img,model_dir,m):

	a = torch.load(model_dir)

    
	obj1 = model_pip(model_in=m,data_path = '/data/gabriel/pat_acc/temp/',
	                         batch_size=1, gpu=0,scale = True,use_gpu=True,resume=False,verbose=False)

	#obj1.cuda()
	obj1.load_model(model_dir)

	L=1
	inp = img
	count = 0 
	name = []
	d_layer = 0
	per_l = 0
	name_i = []

	
	
	for i in obj1.model.children():
		name_i.append(str(i))
		name_i.append('\n')
		

		
		if(isinstance(i,nn.Sequential)):
			#print('here')
			#print(i)
			with open('name.txt','w') as f:
				f.write(str(i))
			for j in i.children():
			#for j,z in i.named_parameters():
				print(type(j))
				#if(isinstance(j,models.densenet._DenseBlock)):
					#print(j)
					#print(z)
				#print(j)
				#print(count)
				#adding layers
				# if(isinstance(j,nn.Sequential)):
				# 	print('here')
				# 	print(j)
				if(isinstance(j,models.densenet._DenseBlock)):
					#print(j)
					for k in j.children():
						d_layer+=1
						for p in k.children():
							per_l+=1
							#print(p)
							#print('\n\n')

					#break
					print(d_layer)
					print(per_l)
					exit()
				with open('name_i.txt','a') as f:
					f.write(str(j))

				name.append(str(j))
				name.append('\n')

				# if(isinstance(j,models.densenet._Transition)):
				# 	print('h')
				# 	print(j)
				# 	print(count)


			count+=1
	name = str(name)
	name_i = str(name_i)
	
		#print(j)
	#tempm = nn.Sequential(*list(obj1.model.children())[L-1])(inp)
	#print(tempm)
	# for i,j in obj1.model.named_parameters():
	# 	print(i)
	# print(obj1.model(img))



import scipy.misc
i = plt.imread('/data/gabriel/OCR/test1.png')
i = scipy.misc.imresize(i,(300,300)).astype(float)
print(i.shape)
i.reshape(1,1,i.shape[0],i.shape[1])
print(i.shape)
j = np.zeros((1,3,300,300))
j[0,0,:,:] = i
j[0,1,:,:] = i
j[0,2,:,:] = i
j = torch.Tensor(j)
j = j.view(1,3,300,300)
print(j.shape)
i = Variable(j)
model_prim = models.densenet161(pretrained=False)

model_prim.classifier = nn.Linear(model_prim.classifier.in_features,7)

#print(model_prim(i).max())

#model_prim.modules()['']

s_points(i,'/data/gabriel/VC_1/SET5/DenseNetModel_pretrained_7_views_bs_64_e_50_26092017_182851.pth.tar',model_prim)

