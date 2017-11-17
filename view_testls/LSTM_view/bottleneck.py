### generate bottlenecks
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
import Augmentor
import pickle
import shutil
data_transform = transforms.Compose([transforms.Scale(300),
	transforms.CenterCrop(300),
	transforms.ToTensor(),
	transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

d_net = models.densenet161(pretrained=True)


class gen_b(object):
	def __init__(self,model,data_dir,save_dir,b_size=4):
		self.b_size= b_size
		self.save_dir = save_dir
		self.model = model.eval()
		self.data_dir = data_dir
		self.data_transform = transforms.Compose([transforms.Scale(300),
												transforms.CenterCrop(300),
												transforms.ToTensor(),
												transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
		self.classes = [i for i in os.listdir(self.data_dir+'/test/') if '.' not in i]
		
	def get_data(self,fol,cl_name):
		cl = [i for i in os.listdir(self.data_dir+'/'+fol) if '.' not in i]
		if(not(os.path.exists(self.save_dir))):
			#shutil.rmtree(self.save_dir)
			os.makedirs(self.save_dir)
		if(not(os.path.exists(self.save_dir+'/'+fol))):
			os.makedirs(self.save_dir+'/'+fol)
		
		if(not(os.path.exists(self.save_dir+'/'+fol+'/'+cl_name))):
			print("acquiring data")
			os.makedirs(self.save_dir+'/'+fol+'/'+cl_name)
			
			data_loc_dic = {i:os.listdir(self.data_dir+'/'+fol+'/'+i) for i in cl}

			count_img_class = {i:len(data_loc_dic[i]) for i in data_loc_dic}
			num_img = 0
			for i in count_img_class:
				num_img+=count_img_class[i]

			label_dic = {class_name:count for count,class_name in zip(cl,np.arange(0,len(cl)))}
			name_num_table = {}
			
			
			
		
			#print(data_loc_dic[i])
			num_img = count_img_class[cl_name]
			img_data = torch.zeros((num_img,3,300,300))
			name_data = torch.zeros(num_img)
			count = 0
			for im_name in data_loc_dic[cl_name]:
				img = self.data_transform(Image.open(self.data_dir+'/'+fol+'/'+cl_name+'/'+im_name))

				img_data[count,:,:,:] = img
				name_data[count] = count
				name_num_table[count] = im_name
				count+=1

			dset = {fol:torch.utils.data.TensorDataset(img_data,name_data)}
			dset_loader ={fol:torch.utils.data.DataLoader(dset[fol],batch_size=self.b_size,shuffle=False,num_workers=4)}
			#torch.save(dset,open(self.save_dir+'/'+fol+'/'+cl_name+'/dataset_'+cl_name+'_'+fol+'.pth','wb'))
			torch.save(dset_loader,open(self.save_dir+'/'+fol+'/'+cl_name+'/dataset_loader_'+cl_name+'_'+fol+'.pth','wb'))
			pickle.dump(name_num_table,open(self.save_dir+'/'+fol+'/name_data_'+fol+'.pkl','wb'))
		
			#dset = {fol:torch.utils.data.TensorDataset(img_data,name_data)}
			#dset_loader ={fol:torch.utils.data.DataLoader(dset[fol],batch_size=self.b_size,shuffle=False,num_workers=4)}
			#torch.save(dset,open(self.save_dir+'/'+fol+'/dataset_'+fol+'.pth','wb'))
			#torch.save(dset_loader,open(self.save_dir+'/'+fol+'/dataset_loader_'+fol+'.pth','wb'))
			print("data acquired for class "+cl_name)
			return dset_loader
			
		else:
			
			dset_loader=torch.load(self.save_dir+'/'+fol+'/'+cl_name+'/dataset_loader_'+cl_name+'_'+fol+'.pth')
			print("data acquired for class "+cl_name)
			return dset_loader
			
	def get_f(self):
		for i,j in d_net.named_children():
			if(i == 'features'):
				new_model = j
		
		new_model.cuda()

		for phase in ['train','val','test']:
			print(phase)
			for class_id in self.classes:
				dataset_loader = self.get_data(phase,class_id)
				num_to_name = pickle.load(open(self.save_dir+'/'+phase+'/name_data_'+phase+'.pkl','rb'))
				#print(num_to_name)
				print("generating bottleneck")
				for data in dataset_loader[phase]:
					#print(data)
					img_batch,name_batch = data
					img = Variable(img_batch.cuda())
					out = new_model(img)
					feat = out
					#print(out.size())
					out = F.relu(feat,inplace=True)
					out = F.avg_pool2d(out,kernel_size=7,stride=1).view(feat.size(0),-1)
					#out = out.view(self.b_size,-1)
					out_save = out.cpu().data.numpy()
					name_batch1 = name_batch.numpy()
					for i in range(0,name_batch1.shape[0]):
						#print(name_batch[i])
						#print(type(i))
						torch.save(out_save[int(i),:],self.save_dir+'/'+phase+'/'+class_id+'/'+str(num_to_name[name_batch1[int(i)]])+'.pth')
						
					#print(img_batch.size())
					del(out)
					del(img)
					del(out_save)
					del(img_batch)
					del(name_batch)
ab = gen_b(model = models.densenet161(pretrained=True),data_dir='/data/gabriel/VC_1/SET7/dataset/',save_dir='/data/gabriel/bottleneck_codes/',b_size= 12)
ab.get_f()

from nndev import model_pip
res = models.densenet161(pretrained=True)
res.classifier = nn.Linear(res.classifier.in_features,7)


obj1 = model_pip(model_in=res,data_path = '/data/gabriel/VC_1/SET7/dataset/',batch_size=1, gpu=0,scale = True,use_gpu=True,resume=False,verbose=False)
obj1.load_model('/data/gabriel/VC_1/SET7/DenseNetModel_pretrained_7_views_bs_64_e_50_26092017_182851.pth.tar')
			
ab2 = gen_b(model = obj1.model ,data_dir='/data/gabriel/VC_1/SET7/dataset/',save_dir='/data/gabriel/bottleneck_codes_echo_pre/',b_size= 12)
ab2.get_f()

		