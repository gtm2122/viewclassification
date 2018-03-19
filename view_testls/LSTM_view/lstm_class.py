import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import torch.optim as optim
import torch.nn.functional as F

import random
#torch.nn.Module.dump_patches=True
### This script is essentially the neural network architecture along with wrappers for training and testing
import re



class lstm_proc(nn.Module):
	def __init__(self,window_len=0,data_dir=0,cache_dir=0,overwrite=0,epochs=10,hidden_dim = 1000,num_views=15,
		embed_sz=19872,layers=1,dropout = 0,batch_size=1,lr = 0.005,gpu_num=0,step_lr = [50]):
		super(lstm_proc,self).__init__()
		self.epochs=epochs
		if(window_len==0):
			self.window_len = window_len-1
		else:
			self.window_len=window_len
		self.lr = lr
		self.step_lr = step_lr
		self.cache_dir = cache_dir
		self.hidden_dim=hidden_dim
		self.gpu_num = gpu_num
		self.num_views=num_views
		self.embed_sz = embed_sz
		self.data_dir = data_dir
		self.batch_size=int(batch_size)
		self.ow = overwrite
		self.labels = [i for i in os.listdir(self.data_dir+'/test/') if '.p' not in i]
		self.loss_fn = nn.NLLLoss()
		self.layers = layers
		self.dropout = dropout
		self.lstm = nn.LSTM(input_size = self.embed_sz,hidden_size = self.hidden_dim,num_layers = self.layers, dropout = self.dropout)
		self.cl = nn.Linear(self.hidden_dim*self.window_len,self.num_views)
		self.hidden = self.init_hidden()
		self.lab_to_ix = {x:i for i,x in enumerate(self.labels)}
		self.ix_to_lab = {self.lab_to_ix[i]:i for i in self.lab_to_ix.keys()}
	# def init_hidden(self):
	# 	return (Variable(torch.zeros(1,1,self.hidden_dim)),Variable(torch.zeros(1,1,self.hidden_dim)))

	def forward(self,x):
		out1,self.hidden = self.lstm(x.view(self.window_len,self.batch_size,self.embed_sz),self.hidden)
		#print(out1)
		#print(out1[-1])
		#exit()
		#exit()
		#print(out1.size())
		#print(lex(x))
		#print(F.log_softmax(self.cl(out1.view(len(x),-1)),dim=0))
		#print(F.log_softmax(self.cl(out1).view(len(x),-1),dim=1))
		#exit()
		#print(self.cl(out1))
		# print(x)
		# print(out1.size())
		# print(x.size())
		# print(out1.permute(1,0,2).size())
		# print(len(x))
		# print(self.hidden_dim)
		# print(out1.permute(1,0,2).view(-1,self.hidden_dim*len(x)))
		input_to_cl1 = out1.permute(1,0,2).cpu().data
		#print(input_to_cl1.size())
		input_to_cl1 = Variable(input_to_cl1.contiguous().cuda())
		return F.log_softmax(self.cl(input_to_cl1.view(self.batch_size,self.hidden_dim*len(x))),dim=1)

		

	def init_hidden(self):
		return (Variable(torch.zeros(self.layers,self.batch_size,self.hidden_dim)),Variable(torch.zeros(self.layers,self.batch_size,self.hidden_dim)))
	@staticmethod
	def find_max_frame(phase_path,phase):
		
		
		if(os.path.exists('./'+phase+'_max_num.pkl')):
			return pickle.load(open('./'+phase+'_max_num.pkl','rb'))
		else:
			leng_max = 0 
			for i in os.listdir(phase_path):
				for j in os.listdir(phase_path+'/'+i):

					if(len(os.listdir(phase_path+'/'+i+'/'+j))>len_max):
						leng_max = len(os.listdir(phase_path+'/'+i+'/'+j))
			pickle.dump(leng_max,open('./'+phase+'_max_num.pkl','wb'))
			return leng_max


	def load_data(self, phase, step_size=1, ow=False, truncate=True,dir_save_path = '/home/gam2018/cached_dsets/img_paths/100_pat_dset',use_self_batch=True,batch_size=100):

		assert step_size > 0 and isinstance(step_size, int), "step size must be a whole number"
		###
		# data_dir = the dataset containing "test","train" etc. The contents of the folder are video folers containing
		# the CNN features extracted for each frame
		# phase = "test"/"train"/"val"
		# ow = overwrite list of paths ?
		# Truncate = Should the batches be limited to having whole number multiple of batch size or not? If not then we must modify the
		# loss function to not include the tail end zeros.
		class_list = self.labels
		seq_len = self.window_len  # number of images in a sequence
		if(use_self_batch):
			batch_size = self.batch_size  # number of sequences in a batch

		embed_sz = self.embed_sz
		if not (os.path.isfile(dir_save_path + '/' + str(seq_len) + '_' + phase + '.pkl')) or ow:
			print("generating new list of image paths")
			img_path_list = []

			for label in class_list:
				for vid_folder in os.listdir(self.data_dir + '/' + phase + '/' + label):
					vid_path = self.data_dir + '/' + phase + '/' + label + '/' + vid_folder
					num_frames = len(os.listdir(vid_path))
					rem_imgs = num_frames % seq_len
					c = 0
					# list_of_imgs = os.listdir(vid_path)
					list_of_imgs = sorted(os.listdir(vid_path),
										  key=lambda s: [int(t) if t.isdigit() else t.lower() for t in
														 re.split('(\d+)', s)])
					# print(list_of_imgs)
					# exit()
					while c <= num_frames - seq_len:
						seq_imgs_names = list_of_imgs[c:c + seq_len]
						seq_imgs_names = [vid_path + '/' + i for i in seq_imgs_names]
						img_path_list.append((seq_imgs_names, label, c))
						c += 1  ## using step size 1 to get all possible windows
					### The above list will contain all possible windows in a list
					### Storing 'c' to enable step size filtering - for eg if c%step_size==0 then include in batch.

			#print(img_path_list[-1])

			pickle.dump(img_path_list, open(dir_save_path + '/' + str(seq_len) + '_' + phase + '.pkl', 'wb'))
		else:
			img_path_list = pickle.load(open(dir_save_path + '/' + str(seq_len) + '_' + phase + '.pkl', 'rb'))
			if (step_size > 1):
				new_path_list = [i for i in img_path_list if i[2] % step_size == 0]
			else:
				new_path_list = img_path_list
			del img_path_list
			total_num_seq = len(new_path_list)
			### Now that we loaded all the img paths for each sequence, we arrange various sequences into a minibatch
			if (truncate):
				#print(len(new_path_list))
				total_num_batches = (total_num_seq // batch_size)
				new_path_list = new_path_list[:total_num_batches * batch_size]
				#print(len(new_path_list))

			left_ind = 0

			right_ind = min(total_num_seq, left_ind + batch_size)

			random.shuffle(new_path_list)
			while left_ind < len(new_path_list):
				### Two ways to deal with the last k sequences, one, truncate to multiple of batch_size, two, zero pad.
				### Trying zero pad.
				#print(left_ind)
				#print(right_ind)
				mini_batch_paths = new_path_list[left_ind:right_ind]
				### loop to load the images into tensors
				img_minibatch = torch.zeros((seq_len, batch_size, embed_sz))
				lab_minibatch = torch.zeros((batch_size))

				mini_batch_idx = 0
				for sequence_tuple in mini_batch_paths:
					### Going through the 100 sequences
					# print(sequence_tuple[0][0])
					frame_num = 0
					for img_paths in sequence_tuple[0]:
						### Now going through the paths of the images in a sequence to fill in the frames of a sequence

						img_minibatch[frame_num, mini_batch_idx, :] = torch.from_numpy(torch.load(img_paths))
						frame_num += 1
					lab_minibatch[mini_batch_idx] = self.lab_to_ix[sequence_tuple[1]]

					mini_batch_idx += 1

				left_ind += batch_size
				right_ind = min(left_ind + batch_size, len(new_path_list))

				### if Truncate set to false then we should use the indices until right_ind for the loss function
				### TODO Set option for truncate = False, return len(list)%batch_size to be used for loss function averaging.
				yield img_minibatch, lab_minibatch.long(),sequence_tuple[0]
			#img_tensor = torch.zeros(max_f,)
			
			#for i in os.listdir(self.data_dir):


from torch.optim import lr_scheduler

def find_2(s):
	return [i for i,j in enumerate(s) if j=='_'][1]



def train_net(model_cl,multi=0):

	criterion = nn.NLLLoss().cuda()

	if(multi==1):

		model_cl = torch.nn.DataParallel(model_cl,dim=1,device_ids = [0,1,2,3,4,5,6,7]).cuda()
	else:
		model_cl = model_cl.cuda()

	op = optim.SGD(model_cl.parameters(),lr=model_cl.lr)
	all_loss_train = []
	all_loss_val = []
	num_to_lab = {}
	#print(model_cl.epochs)
	epoch_list=[]
	best_val_acc = 0

	exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer=op,milestones=model_cl.step_lr,gamma=0.1)

	if(model_cl.batch_size>1):
		agg_pred_dic = {}  # This dictionary is for aggregating per batch results 
	count_train_val = 0

	for epoch in range(0,model_cl.epochs):
	#for epoch in range(0,1):
	
		running_loss_train = 0
		running_loss_val = 0
		conf_m_batch = np.zeros((model_cl.num_views,model_cl.num_views))
		print('model = ', 'layers_'+str(model_cl.layers)+'_hd_'+str(model_cl.hidden_dim)+'_dr_'+str(model_cl.dropout)+'batch_'+str(model_cl.batch_size)+'_loss.png')
		print('epoch = ',epoch)
		# for phase in ['train','train_distort_2_16','val','val_distort_2_16']:
		# for phase in ['val']:
		train_loader = model_cl.load_data(phase='train',step_size=2)
		val_loader = model_cl.load_data(phase='val',step_size=2)
		data_dic ={'train':train_loader,'val':val_loader}
		for phase in [ 'train','val']:

			print(phase)
			#if('2_16' in phase):
			#	print(data_keys[phase])
			if('train' in phase):
				exp_lr_scheduler.step()
				model_cl.train(True)
			else:
				model_cl.train(False)
			data_point_counter = 0
			running_accuracy=0
			for tensor,label,names in data_dic[phase]:
				#print(keys)
				#print(data_keys[phase])

				model_cl.zero_grad()
				if(multi==1):
					model_cl.hidden = (model_cl.init_hidden()[0].cuda(async=False),model_cl.init_hidden()[1].cuda(async=False))
				else:
					model_cl.hidden = (model_cl.init_hidden()[0].cuda(),model_cl.init_hidden()[1].cuda())

				if('val' in phase):
					embeds = Variable(tensor.cuda(),volatile=True)
					label = Variable(label.cuda(),volatile = True)
				else:
					embeds = Variable(tensor.cuda())
					label = Variable(label.cuda())
				out = model_cl(embeds)

				loss = criterion(out.view(out.size(0),-1),label)
				#print(loss)
				if('train' in phase):
					loss.backward()
					running_loss_train+= loss.cpu().data.numpy()[0]
					op.step()
				if('val' in phase):
					### TODO modify accuracy function to get the video wise accuracy,right now use batch wise accuracy.

					running_loss_val+= loss.cpu().data.numpy()[0]

					pred_img,pred_img_idx = out.max(dim=1)

					count=0

					#print(pred_img_idx[:10])
					#print(label.cpu().data[:10])
					#print(names[:10])
					#### conf_m_img is the confusion matrix based on the prediction of a sequence, therefore accuracy will be sequence wise accuracy, not image wise.

					running_accuracy += torch.sum(torch.eq(pred_img_idx.cpu().data,label.cpu().data))
					data_point_counter +=model_cl.batch_size
					# print(pred_img_idx.cpu().data.size())
					# print(label.cpu().data.size())
					# print(running_accuracy)
					# exit()
					for pred_img_val in pred_img_idx.cpu().data.numpy().astype(int):
						#print(pred_img_val)
						conf_m_batch[label.cpu().data[count],pred_img_val]+=1
						count+=1
				del(out)
				del(loss)
				del(label)
				del(embeds)

			if(epoch not in epoch_list):
				epoch_list.append(epoch)
			if('train' in phase):

				all_loss_train.append(running_loss_train)
				count_train_val=0
				plt.plot(epoch_list,all_loss_train)
				if(model_cl.batch_size==1):
					plt.savefig('/data/gabriel/train_loss_rnn/all_data_train_bneck_pre_l'+str(model_cl.layers)+'_hd_'+str(model_cl.hidden_dim)+'_dr_'+str(model_cl.dropout)+'batch_'+str(model_cl.batch_size)+'_loss.png')
				else:
					plt.savefig('/data/gabriel/train_loss_rnn/batch/all_data_train_bneck_pre_l'+str(model_cl.layers)+'_hd_'+str(model_cl.hidden_dim)+'_dr_'+str(model_cl.dropout)+'batch_'+str(model_cl.batch_size)+'_loss.png')
				plt.close()
				#pickle.dump(all_loss_val,open('/data/gabriel/train_val_loss_rnn/train_val_loss_rnn/new_val_hidden_dim_'+str(model_cl.layers)+'_loss.pkl','wb'))
			else:
				# if(count_train_val==0):
				# 	#all_loss_val.append(running_loss_val)
				# 	count_train_val+=1
				# else:
				all_loss_val.append(running_loss_val)
				#vid_acc = conf_m_vid.diagonal().sum()/conf_m_vid.sum()
				print('sequence accuracy =',running_accuracy/data_point_counter)
				img_acc = conf_m_batch.diagonal().sum()/conf_m_batch.sum()
				#print('video accuracy =',vid_acc)
				print('sequence accuracy =',img_acc)
				count_train_val=0
				#print(epoch_list)
				#print(all_loss_val)
				if(img_acc >= best_val_acc):
					best_val_acc = img_acc
					best_model = model_cl

				plt.plot(epoch_list,all_loss_val)
				if(model_cl.batch_size==1):
					plt.savefig('/data/gabriel/train_loss_rnn/all_data_val_bneck_pre_l'+str(model_cl.layers)+'_hd_'+str(model_cl.hidden_dim)+'_dr_'+str(model_cl.dropout)+'batch_'+str(model_cl.batch_size)+'_loss.png')
				else:
					plt.savefig('/data/gabriel/train_loss_rnn/batch/all_data_val_bneck_pre_l'+str(model_cl.layers)+'_hd_'+str(model_cl.hidden_dim)+'_dr_'+str(model_cl.dropout)+'batch_'+str(model_cl.batch_size)+'_loss.png')

				plt.close()

	pickle.dump(all_loss_val,open('all_data_val_'+str(model_cl.layers)+'_'+str(model_cl.lr)+'_loss.pkl','wb'))


	pickle.dump(all_loss_train,open('all_data_train_'+str(model_cl.layers)+'_'+str(model_cl.lr)+'_loss.pkl','wb'))


	return model_cl,op#,num_to_lab
import os


from scipy import stats
import pickle
def test(model_cl,res_dir,test_class='test'):

	if(not os.path.exists(res_dir)):
		os.makedirs(res_dir)
	#os.environ['CUDA_VISIBLE_DEVICES']=str(model_cl.gpu_num)
	test_dic = {str(test_class):model_cl.load_data(phase=test_class)}
	#test_data_keys = list(test_dic[test_class].keys())
	model_cl.eval()
	model_cl = model_cl.cuda()
	
	img_acc = 0
	vid_acc = 0
	total_im = 0
	total_vid = 0


	conf_m_vid = np.zeros((model_cl.num_views,model_cl.num_views))
	#conf_m_img = np.zeros_like(conf_m_vid)

	misc_cl = {}
	agg_pred_dic = {}

	for img_tensor,label_tensor,name_of_imgs in test_data_keys:
		#print(keys)
		#embeds,label,label_name,name_file = test_dic[test_class][keys]

		embeds = Variable(img_tensor.cuda())
		label = Variable(label_tensor.cuda())

		out = model_cl(embeds)

		out_cpu = out.cpu()
		_,pred_lab = torch.max(out_cpu.data,1)

		# pred_lab and label_tensor are the predictions and ground truth for the batch_size

		for i in range(0,pred_lab.size(0)):

			### Sequence accuracy.
			conf_m_img[label.numpy().astype(int)[i],pred_lab[i]]+=1
			actual_lab = pred_lab[i]
			seq_pred = label.numpy().astype(int)[i]
			if(vid_pred!=actual_lab):
				# print(pred_vid[0])
				# print(ac_label)
				misc_cl[tuple(name_of_imgs)] = [model_cl.labels[vid_pred],model_cl.labels[int(actual_lab)]]

	m_name = res_dir[res_dir.find('s_lstm'):res_dir.find('.pth')]
	#print(res_dir)
	#print(m_name)
	pickle.dump(conf_m_vid,open(res_dir+'/'+m_name+'_'+test_class+'_video_conf_m.pkl','wb'))
	pickle.dump(conf_m_img,open(res_dir+'/'+m_name+'_'+test_class+'_image_conf_m.pkl','wb'))
	pickle.dump(misc_cl,open(res_dir+'/'+m_name+'_'+test_class+'_misc_videos.pkl','wb'))

	print(conf_m_vid)
	
	print(conf_m_img)
	print(misc_cl)

	#print('video accuracy = ',conf_m_vid.diagonal().sum()/conf_m_vid.sum())
	print('image accuracy = ',conf_m_img.diagonal().sum()/conf_m_img.sum())
	
