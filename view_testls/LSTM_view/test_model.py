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

def make_loader(sub_dir,b_size=0):
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
	sorted_frame_names = sorted(os.listdir(sub_dir),key = lambda s:[int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)',s)])
								

	for i in sorted_frame_names:
		
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
from collections import Counter

def get_top_3(all_labs_list):
	cnt = Counter(all_labs_list)
	unique_preds = list(set(all_labs_list))
	
	l = cnt.most_common(3)
	new_list = [list(i) for i in l]
	#new_list_normed = copy(new_list)
	sum_prob = sum([i[1] for i in new_list])
	new_list_normed = [[i[0]+'_'+str((1000*i[1]/sum_prob//1)/1000)] for i in new_list]
	if(len(unique_preds)<3):
		for i in range(0,3-len(new_list_normed)):
			new_list_normed.append(['nan'])
	return new_list_normed

def predict(main_dir,fe_model_obj,lstm_obj,output_legend,model_name='densenet',num_views=7,batch_size=0,k_size=7,window_len=0):
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
			if(k_size==9):
				bneck_current = torch.zeros(num_images,1,2208)
			else:
				bneck_current = torch.zeros(num_images,1,19872)
			count = 0
			#os.environ['CUDA_VISIBLE_DEVICES']='3'
			#torch.cuda.set_device(3)
			#with(torch.cuda.device(0)):
				#torch.cuda.set_device(0)
			
			for img,name in loader['test']:
			
				
				img = Variable(img.cuda(),volatile=True)

				out = fe_model_obj(img)
				feat = out
				out = F.relu(feat,inplace=True).squeeze()
				#print(out.size())
				out = F.avg_pool2d(out,kernel_size=k_size,stride=1).view(feat.size(0),-1)
				#print(out.size())
				bneck_current[count,0,:] = out.squeeze().cpu().data
				img = img.cpu()
				del(out)
				del(img)
				#print(lstm_model(out))
				#pred[count] = lstm_model(out)
				count+=1
				#torch.cuda.empty_cache()		
			# if('I16' in i):
			# 	print(bneck_current)
			
			
			gc.collect()
			if(batch_size>1):
				num_frames = bneck_current.size(0)
				
				dim0 = int(np.ceil(num_frames/(window_len*batch_size)))
				
				batched_bneck = torch.zeros(dim0,window_len,batch_size,bneck_current.size(2))
				#print(num_frames)
				rem_frames = num_frames
				count = window_len
					
				for index_num in range(0,dim0):
					### If OOM then consider doing this part by part rather than aggregating as 4-D matrix
					for b_num in range(0,batch_size):
						#print(count)
						if(rem_frames>window_len):
							batched_bneck[index_num,:count,b_num,:] = bneck_current[count-5:count,0,:]
							count+=5
							rem_frames -=count

						elif(rem_frames<window_len and rem_frames>0):
							#print(rem_frames)
							#print(count)
							#print(b_num)
							batched_bneck[index_num,:rem_frames,b_num,:] = bneck_current[count-5:count-5+rem_frames,0,:] 
				# if('I16' in i):
				# 	print(num_frames)
				# 	print(window_len*batch_size)
				# 	print(num_frames/(window_len*batch_size))
				# 	print(np.ceil())
				# 	print(dim0)
				# 	print(window_len)
				# 	print(batch_size)

				# 	print(batched_bneck)
				bneck_current_gpu = batched_bneck.cuda()
			
			else:
				bneck_current_gpu = Variable(bneck_current.cuda(),volatile=True)
				


			if(flag==1):
				model_list_preds = []
				c = 0
				#print("LSTM")
				#print("LSTM")
				for nums in range(0,len(lstm_obj)):  ### List of tuples containing name and the model obj
					#print("LSTM")
					try:

						if(batch_size<=1):
							_,arg_max = lstm_obj[nums](bneck_current_gpu).max(dim=1)
							final_prediction = num_to_lab[stats.mode(arg_max.cpu().data.numpy())[0][0]]
						
							#torch.cuda.empty_cache()
							all_preds1 = [num_to_lab[i] for i in list(arg_max.cpu().data)]
							final_prediction = get_top_3(all_preds1)
							model_list_preds+=[final_prediction]
							#del(pred)
							del(arg_max)
							#print(model_list_preds)
						else:
							final_pred = []
							for index_num in range(0,dim0):
								# print(bneck_current_gpu.size())
								# print(bneck_current_gpu[index_num,:,:,:].size())
								# print(lstm_obj[nums](bneck_current_gpu[index_num,:,:,:]))
								
								# print(lstm_obj[nums](bneck_current_gpu[:,:,:,:]))
								# print(lstm_obj[nums](bneck_current_gpu[index_num,:,:,:]))
								# if('I16' in i):
								# 	print(bneck_current_gpu[index_num,:,:,:])
								_,arg_max = lstm_obj[nums](Variable(bneck_current_gpu[index_num,:,:,:],volatile=True)).max(dim=1)
								final_pred+=list(arg_max.cpu().data)
								del(arg_max)

							all_preds1 = [num_to_lab[i] for i in final_pred]
							#final_prediction = num_to_lab[stats.mode(final_pred)[0][0]]
							
							final_prediction = get_top_3(all_preds1)

							model_list_preds+=[final_prediction,'']
						
							#print(model_list_preds)
						

					except Exception as e:
						print(e)
						if batch_size<=1 :
							err_log_name = 'errorlog.txt'
						else:
							err_log_name = 'batch_errorlog.txt'
						with open(err_log_name,'a') as fi:
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
					#torch.cuda.empty_cache()
				#print(model_list_preds)

				#torch.cuda.empty_cache()
				#bneck_current_gpu = bneck_current_gpu.cpu()

				del(bneck_current_gpu)
				
				
				# print("all_preds)")
				# print(model_list_preds)
				# print(i)
				# print(all_preds)
				all_preds.append([i[find_l2(i):]])
				#print(all_preds)
				all_preds[-1]+=model_list_preds

				# print(i[find_l2(i):])
				
				# print(all_preds)
			else:
				pred,arg_max = lstm_model(Variable(bneck_current.cuda(),volatile=True)).max(dim=1)
				#print(arg_max)
				final_prediction = num_to_lab[stats.mode(arg_max.cpu().data.numpy())[0][0]]
				all_preds.append([i,final_prediction])
				
				#break
	#print(all_preds)
	return all_preds
	#pickle.dump(l_dic,open(main_dir+'res.pkl','wb'))

# if __name__=="__main__":

# 	model_names = ['/data/gabriel/saved_lstm//lstm_hd1500_layers_1_dr_0','/data/gabriel/saved_lstm//lstm_hd1000_layers_1_dr_0','/data/gabriel/saved_lstm//lstm_hd1500_layers_2_dr_0',] ## Top 3
# 	#model_names = '/data/gabriel/saved_lstm//lstm_hd1500_layers_2_dr_0'#] ## Top 3

# 	model_obj_list = [torch.load(i+'.pth').eval().cuda() for i in model_names]

# 	fe = get_extractor('/data/gabriel/VC_1/SET7/DenseNetModel_pretrained_7_views_bs_64_e_50_26092017_182851.pth.tar').eval().cuda()
# 	if batch_size<=1 :
# 		err_log_name = 'errorlog.txt'
# 	else:
# 		err_log_name = 'batch_errorlog.txt'

# 	with open(err_log_name,'w') as fi:
# 		fi.write('___________________________________ERROR LOG________________________________')

# 	output_tr = {0:'PSA',5:'PLA',6:'A3C',3:'A4C',4:'A2C',1:'SCX',2:'SSX'}
# 	result = []
# 	for sub_f in os.listdir('/data/Gabriel_done_deid/G4_imgs/'):

# 		result+=(predict(main_dir='/data/Gabriel_done_deid/G4_imgs/'+sub_f,fe_model_obj=fe,lstm_obj=model_obj_list,output_legend=output_tr,model_name='densenet',num_views=7))	
	
# 	print(result)
# 	with open('./G4_lstm_results.csv','w') as f:
# 		writer = csv.writer(f)
# 		writer.writerows(result)
