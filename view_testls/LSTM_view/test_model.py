import matplotlib.pyplot as plt
import  torch.utils.data as data_utils
import torch.nn as nn
#from torchvision.models import inception
#from torchvision.models import Inception3

from torch.autograd import Variable
import torch.nn.functional as F

from torchvision import datasets,models,transforms
import torch.optim as optim
import random
import gc
import os
import copy
import torch
import numpy as np
from torch.autograd import Variable
import sys
import os
import scipy
import cv2
from PIL import Image
#import Augmentor
import pickle
import shutil

## Generate bottleneck for given dicom file converted to images using matlab
img_tf = transforms.Compose([transforms.Scale(300),
				transforms.CenterCrop(300),
				transforms.ToTensor(),
				transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])


def get_extractor(model_dir,num_views=7,model_name='densenet'):
	res = models.densenet161(pretrained=False)

	## todo ? try a lighter model like resnet for feature extraction

	res.classifier = nn.Linear(res.classifier.in_features,num_views)
	obj1 = model_pip(model_in=res,data_path = '.',batch_size=1, gpu=0,scale = True,use_gpu=True,resume=False,verbose=False)
	obj1.load_model('/data/gabriel/VC_1/SET7/DenseNetModel_pretrained_7_views_bs_64_e_50_26092017_182851.pth.tar')
	if(model_name=='densenet'):
		f_extract = nn.Sequential()
		for i,j in obj1.model.named_children():
			if(i=='features'):
				f_extract=j

	return f_extract.cuda()


def make_loader(sub_dir):
	## input in this is the element in 'fols' generator ie i in os.listdir(main_dir)
	## This function makes data into loaders for faster pytorching 
	num_images = (sum(1) for i in os.listdir(sub_dir) if '.' not in i[0])
	img_tensor = torch.zeros(num_images,3,300,300)
	name_data = torch.zeros(num_images)
	num_to_name = {}
	count=0
	for i in os.listdir(sub_dir):
		if('.png' in i):
			img_tensor[count,:,:,:] = img_tf(Image.open(i))
			name_data[count]=count
			num_to_name[count]=i

	dset = {'test':torch.utils.data.TensorDataset(img_tensor,name_data)}

	dset_loader ={'test':torch.utils.data.DataLoader(dset[fol],batch_size=1,shuffle=False)}

	return dset_loader

def make_bneck(main_dir,model_obj,model_name='densenet',num_views=7):
# This will make a loader for the bottlenecks of the images in a particular dicom file
	fols = (main_dir+'/'+i for i in os.listdir(main_dir) '.' not in i[0]) # This should give the name of the dicom file
	
	fe = model_obj

	for i in fols:
		loader = make_loader(i)
		
		for img,name in loader['test']:

			img = img.cuda()

			out = fe(img)
			feat = out
			out = F.relu(feat,inplace=True)

			out = F.avg_pool2d(out,kernel_size=7,stride=1).view(feat.size(0),-1)


