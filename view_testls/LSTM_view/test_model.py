import matplotlib.pyplot as plt
import  torch.utils.data as data_utils
import torch.nn as nn
#from torchvision.models import inception
#from torchvision.models import Inception3
import sys
sys.path.append('../')
from nndev import model_pip
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
from lstm_class import *
## Generate bottleneck for given dicom file converted to images using matlab
img_tf = transforms.Compose([transforms.Resize((300,300)),
				transforms.CenterCrop(300),
				transforms.ToTensor(),
				transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])


def get_extractor(model_dir,num_views=7,model_name='densenet'):
	res = models.densenet161(pretrained=False)

	## todo ? try a lighter model like resnet for feature extraction

	res.classifier = nn.Linear(res.classifier.in_features,num_views)
	obj1 = model_pip(model_in=res,data_path = '/data/gabriel/VC_1/SET7/dataset/',batch_size=1, gpu=0,scale = True,use_gpu=True,resume=False,verbose=False)
	#obj1.load_model('/data/gabriel/VC_1/SET7/DenseNetModel_pretrained_7_views_bs_64_e_50_26092017_182851.pth.tar')
	obj1.load_model(model_dir)
	
	if(model_name=='densenet'):
		f_extract = nn.Sequential()
		for i,j in obj1.model.named_children():
			if(i=='features'):
				f_extract=j

	return f_extract

def load_lstm(model_dir):
	return torch.load(model_dir)

def make_loader(sub_dir):
	## input in this is the element in 'fols' generator ie i in os.listdir(main_dir)
	## This function makes data into loaders for faster pytorching
	#if(os.isdir(sub_dir)): 
	num_images = len(os.listdir(sub_dir))
	img_tensor = torch.zeros(num_images,3,300,300)
	name_data = torch.zeros(num_images)
	num_to_name = {}
	count=0
	#print('make loader '+sub_dir+'\n')
	#print(os.listdir(sub_dir))
	for i in os.listdir(sub_dir):
		#print(i)
		#print('\n')
		#print(i)
		if('.bmp' in i):
			#plt.imshow(plt.imread(sub_dir+'/'+i)),plt.show()
			img_tensor[count,:,:,:] = img_tf(Image.open(sub_dir+'/'+i))
			name_data[count]=count
			#print(i)
			num_to_name[count]=i
			count+=1
	dset = {'test':torch.utils.data.TensorDataset(img_tensor,name_data)}
	#print(num_to_name)
	dset_loader ={'test':torch.utils.data.DataLoader(dset['test'],batch_size=1,shuffle=False)}

	return dset_loader,num_images,num_to_name

def find_l2(s):
	return [i for i,j in enumerate(s) if j=='/'][-2]
import csv
import gc
def predict(main_dir,fe_model_obj,lstm_obj,output_legend,model_name='densenet',num_views=7):
# This will make a loader for the bottlenecks of the images in a particular dicom file
# main_dir is /to/studyid/0KXc1UFT/ contains folders for each dicom file, and inside each fodler are .png files
	fols = [main_dir+'/'+i for i in os.listdir(main_dir) if '.' not in i[0]]  
	
	flag = 0

	if(isinstance(lstm_obj,list)):
		flag=1
		# for i in lstm_obj:
		# 	i = i.eval()

	else:

		#fe = fe_model_obj.eval().cuda()
		lstm_model = lstm_obj.eval().cuda()

	num_to_lab = output_legend
	all_preds = []
	#fe= fe.cuda()
	
	for i in fols:
		## Here i is the folder containing the .png images
		#print(i)
		if(os.path.isdir(i)):

			loader,num_images,num_to_name_dic = make_loader(i)
			#print(i)
			pred = torch.zeros(num_images)
			bneck_current = torch.zeros(num_images,1,19872)
			count = 0
			#os.environ['CUDA_VISIBLE_DEVICES']='3'
			#torch.cuda.set_device(3)
			#with(torch.cuda.device(0)):
				#torch.cuda.set_device(0)
			
			for img,name in loader['test']:
				#print(img.size())
				#img2 = img.numpy().squeeze().transpose(1,2,0)
				#scipy.misc.imsave('./abc.png',img2)
				#print(img2.shape)
				#plt.imshow(img2),plt.show()
				
				img = Variable(img).cuda()

				out = fe(img)
				feat = out
				out = F.relu(feat,inplace=True)
				#print(out.size())
				out = F.avg_pool2d(out,kernel_size=7,stride=1).view(feat.size(0),-1)
				bneck_current[count,0,:] = out.cpu().data
				img = img.cpu()
				del(out)
				del(img)
				#print(lstm_model(out))
				#pred[count] = lstm_model(out)
				count+=1
				torch.cuda.empty_cache()		
			#print(count)
			#print(lstm_model(Variable(bneck_current.cuda())))
			#os.environ['CUDA_VISIBLE_DEVICES']='3'
			#torch.cuda.set_device(3)
			
			gc.collect()

		#with torch.cuda.device(1):
			gc.collect()
			#print(flag)
			#print("PRE LSTM")
			#torch.cuda.set_device(2)
			if(flag==1):
				model_list_preds = []
				c = 0
				
				
				# for items in lstm_obj:
				# 	items = items.cuda()

				#print(lstm_obj)
				#lstm_obj_c = lstm_obj.copy()
				c=0
				#print("LSTM")
				bneck_current_gpu = Variable(bneck_current[:min([70,bneck_current.size(0)]),:,:].cuda())
				#print("LSTM")
				for nums in range(0,len(lstm_obj)):  ### List of tuples containing name and the model obj
					#print("LSTM")
					try:
						_,arg_max = lstm_obj[nums](bneck_current_gpu).max(dim=1)
						final_prediction = num_to_lab[stats.mode(arg_max.cpu().data.numpy())[0][0]]
					
						torch.cuda.empty_cache()
						model_list_preds+=[final_prediction]
						#del(pred)
						del(arg_max)
					except Exception as e:
						print(e)
						with open('errorlog.txt','a') as fi:
							fi.write('\n')
							fi.write(str(e))
							fi.write('\n')
							fi.write(str(i))
							fi.write('\n')
							fi.write(num_to_name_dic[name[0]])
							fi.write('\n')
							fi.write(str(bneck_current_gpu.size(0)))
							fi.write('\n')
							fi.write(str(nums))
							fi.write('\n')
							fi.write('________________________________________________________________________________________________')

							model_list_preds+=[np.nan]
							torch.cuda.empty_cache()
					
					gc.collect()
					torch.cuda.empty_cache()
				#print(model_list_preds)

				torch.cuda.empty_cache()
				#bneck_current_gpu = bneck_current_gpu.cpu()

				del(bneck_current_gpu)
				
				all_preds.append([i[find_l2(i):]])

				all_preds[-1]+=model_list_preds
			else:
				pred,arg_max = lstm_model(Variable(bneck_current.cuda())).max(dim=1)
				#print(arg_max)
				final_prediction = num_to_lab[stats.mode(arg_max.cpu().data.numpy())[0][0]]
				all_preds.append([i,final_prediction])
				
				#break
	#print(all_preds)
	return all_preds
	#pickle.dump(l_dic,open(main_dir+'res.pkl','wb'))

if __name__=="__main__":

	model_names = ['/data/gabriel/saved_lstm//lstm_hd1500_layers_1_dr_0','/data/gabriel/saved_lstm//lstm_hd1000_layers_1_dr_0','/data/gabriel/saved_lstm//lstm_hd1500_layers_2_dr_0'] ## Top 3
	#model_names = '/data/gabriel/saved_lstm//lstm_hd1500_layers_2_dr_0'#] ## Top 3

	model_obj_list = [torch.load(i+'.pth').eval().cuda() for i in model_names]
	#model_obj_list = torch.load(model_names+'.pth') 
	#
	# for i in model_obj_list:
	# 	test(i[1],'./bla.pkl')

	fe = get_extractor('/data/gabriel/VC_1/SET7/DenseNetModel_pretrained_7_views_bs_64_e_50_26092017_182851.pth.tar').eval().cuda()
	
	with open('errorlog.txt','w') as fi:
		fi.write('___________________________________ERROR LOG________________________________')

	output_tr = {0:'PSA',5:'PLA',6:'A3C',3:'A4C',4:'A2C',1:'SCX',2:'SSX'}
	result = []
	for sub_f in os.listdir('/data/Gabriel_done_deid/G4_imgs/'):

		result+=(predict(main_dir='/data/Gabriel_done_deid/G4_imgs/'+sub_f,fe_model_obj=fe,lstm_obj=model_obj_list,output_legend=output_tr,model_name='densenet',num_views=7))	
	
	print(result)
	with open('./G4_lstm_results.csv','w') as f:
		writer = csv.writer(f)
		writer.writerows(result)
