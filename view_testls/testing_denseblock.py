import matplotlib.pyplot as plt

import  torch.utils.data as data_utils
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from torchvision import datasets,models,transforms
import torch.optim as optim

import random
import gc

import copy
import torch
from torch.autograd import Variable
import sys
import os
import scipy
import cv2
from PIL import Image
from nndev import model_pip
from collections import OrderedDict
import pickle
### WORKING VERSION, NEEDS MORE WORK
#m = models.densenet161(pretrained=True)

res = models.densenet161(pretrained=True)

res.classifier = nn.Linear(res.classifier.in_features,7)

m = model_pip(model_in=res,data_path = '/data/gabriel/pat_acc/temp/',
                         batch_size=1, gpu=0,scale = True,use_gpu=True,resume=False,verbose=False)

#m2 = models.densenet161(pretrained=False)
m.load_model('/data/gabriel/VC_1/SET7/DenseNetModel_pretrained_7_views_bs_64_e_50_26092017_182851.pth.tar')


class dnet(nn.Module):
	def __init__(self):
		super(dnet,self).__init__()
		self.features = nn.Sequential(OrderedDict([
						('conv0',nn.Conv2d(3,96,kernel_size=7,stride=2,padding=3,bias=False)),
						('norm0',nn.BatchNorm2d(96)),
						('relu0',nn.ReLU(inplace=True)),
						('pool0',nn.MaxPool2d(kernel_size=3,stride=2,padding=1)),
											])
										)

		
	def forward(self,x):
		features = self.features(x)
		return features

m2 = dnet()

#print(m2)
bb = torch.zeros((1,3,300,300))
bb = Variable(bb)
#exit()
#for p in m2.state_dict():
#	print(p)
c = 0
t = 0 
#img = plt.imread('/data/gabriel/')
print(bb.size())

avg_f_maps = {}

print(m2(bb).size())

## For initial convolution
c = []
m.model.eval()
m2.eval()
b = []
for i in m.model.children():
	if(isinstance(i,nn.Sequential)):

		if(isinstance(i,models.densenet._DenseBlock)):
			break

		for pp in i.children():
			#print(pp)
			if(isinstance(pp,models.densenet._DenseBlock)):
				break
			elif(isinstance(pp,nn.Conv2d) ):
				#print(m2.features[pp])
				for kk in m2.modules():
					if(isinstance(kk,nn.Conv2d)):
						#print(kk)
						kk.weight.data=pp.weight.data
						c.append(pp.weight.data.numpy())
						break
			elif(isinstance(pp,nn.BatchNorm2d)):
				for kk in m2.modules():
					if(isinstance(kk,nn.BatchNorm2d) and kk.bias.data.numpy().shape[0]==96 ):
						#print(kk)
						#print(pp.weight.data.size())
						kk.weight.data=pp.weight.data
						kk.bias.data=pp.bias.data
						kk.running_mean=pp.running_mean
						kk.running_var=pp.running_var
						
						b.append(pp.weight.data.numpy())
						b.append(pp.bias.data.numpy())
						b.append(pp.running_mean.numpy())
						b.append(pp.running_var.numpy())

						
		print(m2)
		# for pp in m2.state_dict().keys():
		# 	#print(m2[pp])
		# 	#print(pp)
		# 	if('conv0' in pp):
		# 		print(np.allclose(m2.state_dict()[pp].numpy(),c[0]))
		# 	if('norm0.weight' in pp):
		# 		#print(b[0].shape)
		# 		#print(pp)
		# 		#print(m2.state_dict()[pp].size())
		# 		print(np.allclose(m2.state_dict()[pp].numpy(),b[0]))
		# 	if('norm0.bias' in pp):
		# 		print(np.allclose(m2.state_dict()[pp].numpy(),b[1]))

		# 	if('norm0.running_mean' in pp):
		# 		print(np.allclose(m2.state_dict()[pp].numpy(),b[2]))
		# 	if('norm0.running_var' in pp):
		# 		print(np.allclose(m2.state_dict()[pp].numpy(),b[3]))

avg_f_maps['init'] = m2(bb)

def norm_filt(f,im_up):
	f = im_up.min() + (f - f.min())*(im_up.max()-im_up.min())/(f.max()-f.min())
	return f

s = nn.Sequential(*list(m.model.features.children()))
#print(s[4])
import shutil
def get_fmaps(orig_model,new_model,img,n):
	### Function to get intermediate averaged feature maps
	k = 0
	try:
		
		os.makedirs(n[:n.find('.jpg')])
	except:
		shutil.rmtree(n[:n.find('.jpg')])
		os.makedirs(n[:n.find('.jpg')])

	f_maps = {}
	f_maps_l = []
				
	c = 0
	t = 0
	print(img.size())
	f_maps['input']=img.mean(dim=1).squeeze()

	print(f_maps['input'].max())
	#exit()
	f_maps['initial']=new_model(img).mean(dim=1).squeeze()
	f_maps_l.append(f_maps['input'])
	f_maps_l.append(f_maps['initial'])
	
	plt.imshow(f_maps['input'].squeeze().data.numpy())
	plt.savefig(n[:n.find('.jpg')]+'/input_'+n+'.png')
	plt.close()
	print(f_maps['initial'].size())
	plt.imshow(f_maps['initial'].squeeze(0).data.numpy())
	plt.savefig(n[:n.find('.jpg')]+'/initial_'+n+'.png')
	plt.close()
	#plt.show()
	for i in s:
		#print(i)

		if(isinstance(i,models.densenet._DenseBlock)):
			c+=1
			new_model.features.add_module('denseblock'+str(c),s[4+k])
			k+=1
			f_maps['denseblock'+str(c)]=new_model(img).mean(dim=1).squeeze()
			f_maps_l.append(f_maps['denseblock'+str(c)])
			print(f_maps['denseblock'+str(c)].size())
			plt.imshow(f_maps['denseblock'+str(c)].squeeze(0).data.numpy())
			plt.savefig(n[:n.find('.jpg')]+'/denseblock'+str(c)+'_'+n+'.png')
			plt.close()
			#print(str(i)[:10])
		elif(isinstance(i,models.densenet._Transition)):
			t+=1
			new_model.features.add_module('transition'+str(t),s[4+k])
			f_maps['transition'+str(t)] = new_model(img).mean(dim=1).squeeze()
			k+=1
			print(f_maps['transition'+str(t)].size())
			f_maps_l.append(f_maps['transition'+str(t)])
			plt.imshow(f_maps['transition'+str(t)].squeeze(0).data.numpy())
			plt.savefig(n[:n.find('.jpg')]+'/transition'+str(t)+'_'+n+'.png')
			plt.close()
			#print(str(i)[:10])
	


	new_model.features.add_module('norm5',s[-1])
	del(f_maps['denseblock'+str(c)])
	f_maps['final'] = new_model(img).mean(dim=1)
	f_maps_l.append(f_maps['final'])
	### TODO 
	### Shuold the averaged output be after batch norm or conv2d ?

	#f_maps['final'] = new_model(img).mean(dim=1)
	#print(f_maps['final'].size())
	
	#print(m2)

	with open ('m2.txt','w') as f:
		f.write(str(new_model))
	#m2.features.add_module('denseblock1',)
	#exit()

	f_maps_r = f_maps_l[::-1]

	filt = f_maps_r[0].data.numpy().reshape(9,9)
	print(type(filt))
	print(filt.shape)
	filt = scipy.misc.imresize(filt,(f_maps_r[2].data.numpy().shape[0],f_maps_r[2].data.numpy().shape[1]))

	filt = norm_filt(filt,f_maps_r[2].data.numpy())

	print('asdasd')

	number = 0
	for i in range(2,len(f_maps_r)-2,2):
		
		filt = scipy.misc.imresize(f_maps_r[i].data.numpy()*filt,(f_maps_r[i+2].data.numpy().shape[0],f_maps_r[i+2].data.numpy().shape[1]))
		filt = -1*norm_filt(filt,f_maps_r[i+2].data.numpy())
		plt.imshow(filt)
		print(filt.shape)
		plt.savefig(n[:n.find('.jpg')]+'/'+str(number)+'filter.png')
		plt.close()
		number+=1
	print(filt.shape)
	


	plt.imshow(filt*f_maps['initial'].data.squeeze().numpy())
	#plt.show()
	
	plt.savefig(n[:n.find('.jpg')]+'/'+'initial_layer.png')
	plt.close()

	filt = scipy.misc.imresize(filt*f_maps['initial'].data.squeeze().numpy(),(f_maps['input'].data.numpy().shape[0],f_maps['input'].data.numpy().shape[1]))

	filt = norm_filt(filt,f_maps['input'].data.squeeze().numpy())
	plt.imshow(filt*f_maps['input'].data.squeeze().numpy())
	#plt.show()
	
	plt.savefig(n[:n.find('.jpg')]+'/'+'input_layer.png')
	plt.close()


	return

img_list = ['/data/gabriel/VC_1/SET7/dataset/test/SSX/80_50_53.jpg',
			'/data/gabriel/VC_1/SET7/dataset/test/SCX/105_77_47.jpg',
			'/data/gabriel/VC_1/SET7/dataset/test/PSA/106_10_10.jpg',
			'/data/gabriel/VC_1/SET7/dataset/test/PLA/107_1_6.jpg',
			'/data/gabriel/VC_1/SET7/dataset/test/A4C/106_17_22.jpg',
			'/data/gabriel/VC_1/SET7/dataset/test/A3C/102_31_27.jpg',
			'/data/gabriel/VC_1/SET7/dataset/test/A2C/88_20_33.jpg']

def find_l(word):
	return [i for i,j in enumerate(word) if (j=='/')][-1]+1

for i in img_list:
	bb = Image.open(i)
	#print(np.array(bb).shape)
	data_transforms = transforms.Compose([transforms.Scale(300),
					transforms.CenterCrop(300),
					transforms.ToTensor(),
					transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) ])
	#print(data_transforms(bb).size())
	bb = Variable(data_transforms(bb).view(1,3,300,300))
	name = i[find_l(i):]
	
	m3 = copy.deepcopy(m2)
	get_fmaps(m.model,m3,bb,name)
	del(m3)