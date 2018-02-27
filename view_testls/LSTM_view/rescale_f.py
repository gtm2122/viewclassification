### funciton to reslace LSTM features to [-1,1] to make it compatible for the tanh function inside
import torch
import os
import shutil
import numpy as np
def rescale_f(src,des):
	### path is a directory of folder containing features stored as .pth
	try:
		os.makedirs(des)
	except:
		shutil.rmtree(des)
		os.makedirs(des)
	for img in os.listdir(src):
		out_save = torch.load(src+'/'+img)
		min_values = np.min(out_save)
		max_values = np.max(out_save)
		out_saveN = -1 + 2*(out_save-min_values)/(max_values-min_values)
		torch.save(out_saveN,des+'/'+img)


def rescale_wrapper(main_dir,des_dir):
	### main_dir contains the folder conatining "train"test"val" folders
	try:
		os.makedirs(des_dir)
	except:
		shutil.rmtree(des_dir)
		os.makedirs(des_dir)
	#print(des_dir+'/'+'o')
	for i in os.listdir(main_dir):
		print(i)
		if('test' in i or 'train' in i or 'val' in i):
			if not(os.path.isdir(des_dir+'/'+i)):
				os.makedirs(des_dir+'/'+i)
				
			for j in os.listdir(main_dir+'/'+i):
				### inside main_dir/train for eg
				if ('.p' not in j):
					class_path = main_dir+'/'+i+'/'+j

					if not(os.path.isdir(des_dir+'/'+i+'/'+j)):
						os.makedirs(des_dir+'/'+i+'/'+j)
						
					for k in os.listdir(class_path):
						if(os.path.isdir(class_path+'/'+k)):
							if not(os.path.isdir(des_dir+'/'+i+'/'+j+'/'+k)):
								os.makedirs(des_dir+'/'+i+'/'+j+'/'+k)
							print(class_path+'/'+k,des_dir+'/'+i+'/'+j+'/'+k)
							rescale_f(class_path+'/'+k,des_dir+'/'+i+'/'+j+'/'+k)


rescale_wrapper(main_dir='/data/gabriel/SET1_bnecks/',des_dir='/data/gabriel/SET1_bnecks_normed/')
