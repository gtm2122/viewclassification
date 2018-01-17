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
	def __init__(self,window_len=0,data_dir=0,cache_dir=0,overwrite=0,epochs=10,hidden_dim = 1000,num_views=15,embed_sz=19872,layers=1,dropout = 0,batch_size=1):
		super(lstm_proc,self).__init__()
		self.epochs=epochs
		if(window_len==0):
			self.window_len = window_len-1
		else:
			self.window_len=window_len
		self.cache_dir = cache_dir
		self.hidden_dim=hidden_dim
		self.num_views=num_views
		self.embed_sz = embed_sz
		self.data_dir = data_dir
		self.batch_size=batch_size
		self.ow = overwrite
		self.labels = [i for i in os.listdir(self.data_dir+'/test/') if '.p' not in i]
		self.loss_fn = nn.NLLLoss()
		self.layers = layers
		self.dropout = dropout
		self.lstm = nn.LSTM(input_size = self.embed_sz,hidden_size = self.hidden_dim,num_layers = self.layers, dropout = self.dropout)
		self.cl = nn.Linear(self.hidden_dim,self.num_views)
		self.hidden = self.init_hidden()
	# def init_hidden(self):
	# 	return (Variable(torch.zeros(1,1,self.hidden_dim)),Variable(torch.zeros(1,1,self.hidden_dim)))

	def forward(self,x):
		out1,self.hidden = self.lstm(x.view(self.window_len,self.batch_size,self.embed_sz),self.hidden)
		#print(out1)
		#exit()
		#print(out1.size())
		#print(lex(x))
		#print(F.log_softmax(self.cl(out1.view(len(x),-1)),dim=0))
		#print(F.log_softmax(self.cl(out1).view(len(x),-1),dim=1))
		#exit()
		#print(self.cl(out1))
		return F.log_softmax(self.cl(out1.view(len(x),-1)),dim=1)

		

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


	
	def label_to_ix(self,c):
		#print(self.labels)
		labs = {x:i for i,x in enumerate(self.labels) }
		#print(labs)

		return torch.LongTensor(1).fill_(labs[c])

	def load_data(self,phase):
		#max_f = find_max_frame(self.data_dir+'/'+phase)




		if(os.path.exists(self.cache_dir) and not(self.ow)):
			return torch.load(self.cache_dir)
		else:

			### This is for batch_size = 1 :

			if self.batch_size == 1: ### This refers to num batches, not number in a batch:

				 ### BECAUSE shape is (-1 , 1, 19872)
				#print('here')
				#print(max_f)
				data_dic = {}
				for cl in os.listdir(self.data_dir+'/'+phase+'/'):
					if '.p' not in phase and '.p' not in cl:
						for vid in os.listdir(self.data_dir+'/'+phase+'/'+cl):
							if '.p' not in vid:
								
								if(self.window_len <=0):

									max_f  = len(os.listdir(self.data_dir+'/'+phase+'/'+cl+'/'+vid))
									#print(max_f)

								img_tensor = torch.zeros(max_f,1,self.embed_sz)
								count= 1
								if(len(os.listdir(self.data_dir+'/'+phase+'/'+cl+'/'+vid))>0):
									sorted_frame_names = sorted(os.listdir(self.data_dir+'/'+phase+'/'+cl+'/'+vid+'/'),key = lambda s:[int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)',s)])
									for frame in sorted_frame_names:
										#print(sorted_frame_names)
										img = torch.load(self.data_dir+'/'+phase+'/'+cl+'/'+vid+'/'+frame)
										
										
										img_tensor[count-1,0,:] = torch.from_numpy(img)
										
										count+=1

										if(count>0 and not(count%max_f) and self.window_len>0):
											#print(vid)
											data_dic[vid+str(count)] = (img_tensor,self.label_to_ix(cl),cl)
											img_tensor = torch.zeros(max_f,1,self.embed_sz)
									
									if(vid+str(max_f) not in data_dic):
										data_dic[vid+str(max_f)] = (img_tensor,self.label_to_ix(cl),cl,self.data_dir+'/'+phase+'/'+cl+'/'+vid+'/')		
								
			### This part for non-zero batch size
			else:
				for cl in os.listdir(self.data_dir+'/'+phase+'/'):

					if '.p' not in phase and '.p' not in cl:
						for vid in os.listdir(self.data_dir+'/'+phase+'/'+cl):
							if '.p' not in vid:
								num_frames = len(os.listdir(self.data_dir+'/'+phase+'/'+cl+'/'+vid))
								sorted_frame_names = sorted(os.listdir(self.data_dir+'/'+phase+'/'+cl),key = lambda s:[int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)',s)])
									
								assert self.window_len>=0,"self.window_len should be atleast 0"

								## setting max seq_len*batches = 15
								## minimum num of frames in test/train/val = 13 , calculated using check.largest_folder()
								max_frame_batches = 15


								if(num_frames<max_frame_batches):

									batched_tensor = torch.zeros(self.window_len,self.batch_size,self.embed_sz)

									for num in range(0,self.batch_size):
										count = 0
										for frame in sorted_frame_names[num*self.window_len:(num+1)*self.window_len]:
											batched_tensor[count,num,:] = torch.load(self.data_dir+'/'+phase+'/'+cl+'/'+vid+'/'+frame)
											count+=1

									if(vid+str((num+1)*self.window_len)):
										data_dic[vid+str((num)*self.window_len)] = (batched_tensor,self.label_to_ix(cl),cl,self.data_dir+'/'+phase+'/'+cl+vid+'/')

								else:
									num_batches = np.ceil(num_frames/self.window_len)

									# TODO experiment with zero padding instead of windowing in last frames
									# Posssible TODO - possible image augmentation would be to use moving windows with step=1 rather than step = self.window_len 
									
									num_zeros = num_batches*np.ceil(num_frames/self.window_len) - num_frames

									### Batches are divided into 1:self.window_len , when it reaches the remainder it the batch will comprise of the last (self.window_len) number
									### of frames, with the other two batches being zero vectors. The loss function will be modified to average along this non zero batch only.

									for mega_batch_num in range(0,num_batches//self.batch_size):
										batched_tensor = torch.zeros(self.window_len,self.batch_size,self.embed_sz)
										
										for num in range(0,self.batch_size):
											count=0
											for frame in sorted_frame_names[(num+np.ceil(num_batches//self.batch_size)*mega_batch_num)*self.window_len:(num+1+np.ceil(num_batches//self.batch_size)*mega_batch_num)*self.window_len]:
												batched_tensor[count,num,:] = torch.load(self.data_dir+'/'+phase+'/'+cl+'/'+vid+'/'+frame)
												count+=1

										if(vid+str((num+1)*self.window_len)):
											data_dic[vid+str((num+1)*self.window_len)] = (batched_tensor,self.label_to_ix(cl),cl,self.data_dir+'/'+phase+'/'+cl+vid+'/')

									### taking the last 5 frames and then zero padding rest of the batch
									if(vid+str(num_frames-self.window_len)):
										batched_tensor = torch.zeros(self.window_len,self.batch_size,self.embed_sz)
										count = 0
										for frame in sorted_frame_names[len(sorted_frame_names)-self.window_len:]:
											batched_tensor[count,0,:] = torch.load(self.data_dir+'/'+phase+'/'+cl+'/'+vid+'/'+frame)
											count+=1

										if(vid+str((num+1)*self.window_len)):
											data_dic[vid+str((num+1)*self.window_len)] = (batched_tensor,self.label_to_ix(cl),cl,self.data_dir+'/'+phase+'/'+cl+vid+'/')

								### Computed based on min frames in train_set. I chose value = highest prime factor, but can be increased.



			torch.save(data_dic,self.cache_dir)
			return data_dic
			#img_tensor = torch.zeros(max_f,)
			
			#for i in os.listdir(self.data_dir):


from torch.optim import lr_scheduler




def train_net(model_cl,multi=0):

	all_data_dic = {'train':model_cl.load_data('train'),'val':model_cl.load_data('val')}
	
	
	
	train_data_keys = list(all_data_dic['train'].keys())
	val_data_keys = list(all_data_dic['val'].keys())

	random.shuffle(train_data_keys)
	random.shuffle(val_data_keys)

	data_keys = {'train':train_data_keys,'val':val_data_keys}
	criterion = nn.NLLLoss().cuda()
	#print(data_keys['train'])
	if(multi==1):

		model_cl = torch.nn.DataParallel(model_cl,dim=1,device_ids = [0,1,2,3,4,5,6,7]).cuda()
	else:
		model_cl = model_cl.cuda()

	op = optim.SGD(model_cl.parameters(),lr=0.005)
	all_loss_train = []
	all_loss_val = []
	num_to_lab = {}
	#print(model_cl.epochs)
	epoch_list=[]
	best_val_acc = 0

	exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer=op,milestones=[25],gamma=0.1/3)

	for epoch in range(0,model_cl.epochs):
	#for epoch in range(0,1):
	
		running_loss_train = 0
		running_loss_val = 0
		#print(epoch)
		#epoch_list.append(epoch)
		conf_m_vid = np.zeros((model_cl.num_views,model_cl.num_views))
		conf_m_img = np.zeros_like(conf_m_vid)
		print('model = ', 'layers_'+str(model_cl.layers)+'_hd_'+str(model_cl.hidden_dim)+'_dr_'+str(model_cl.dropout))
		print('epoch = ',epoch)

		for phase in ['train','val']:
		#for phase in ['train','val']:
			#print(phase)
			if(phase=='train'):
				exp_lr_scheduler.step()
				model_cl.train(True)
			else:
				model_cl.train(False)
			for keys in data_keys[phase]:
				#print(keys)
				#print(data_keys[phase])

				model_cl.zero_grad()
				if(multi==1):
					model_cl.hidden = (model_cl.init_hidden()[0].cuda(async=False),model_cl.init_hidden()[1].cuda(async=False))
				else:
					model_cl.hidden = (model_cl.init_hidden()[0].cuda(),model_cl.init_hidden()[1].cuda())
				
				embeds,label,label_name,name_file = all_data_dic[phase][keys]
				if(label not in num_to_lab):
					num_to_lab[label.numpy()[0]] = label_name 
				#print(keys)
				#print(embeds)
				#print(label_name)
				#print(label)
				#print(name_file)
				#print(phase)
				#print(label)
				#print(label.size())
				#print(keys)
				if(model_cl.batch_size==1):
					label = torch.LongTensor(embeds.size(0)).fill_(label[0])
				else:
					label_tensor = torch.zeros(embeds.size(0),embeds.size(1))
					label_tensor = label.fill_(label[0])
					label = label_tensor
								
				embeds = Variable(embeds.cuda())
				label = Variable(label.cuda())
				#print(embeds)
				out = model_cl(embeds)
				#print(out)
				loss = criterion(out.view(out.size(0),-1),label)

				if(phase =='train'):
					loss.backward()
					running_loss_train+= loss.cpu().data.numpy()[0]
					op.step()
				if(phase=='val'):
					running_loss_val+= loss.cpu().data.numpy()[0]

					pred_img,pred_img_idx = out.max(dim=1)
					#print(pred_img_idx)
					#exit()
					pred_vid = stats.mode(pred_img_idx.cpu().data.numpy())[0]
					for pred_img_val in pred_img_idx.cpu().data.numpy().astype(int):
						#print(pred_img_val)
						conf_m_img[label.cpu().data[0],pred_img_val]+=1
					conf_m_vid[label.cpu().data[0],pred_vid]+=1
					#optimizer.step()
					
				del(out)
				del(loss)
				del(label)
				del(embeds)
			if(epoch not in epoch_list):
				epoch_list.append(epoch)
			
			
			if(phase=='train'):
				all_loss_train.append(running_loss_train)
				plt.plot(epoch_list,all_loss_train)
				plt.savefig('/data/gabriel/train_loss_rnn/train_bneck_pre_l'+str(model_cl.layers)+'_hd_'+str(model_cl.hidden_dim)+'_dr_'+str(model_cl.dropout)+'_loss.png')
				plt.close()
				#print(epoch_list)
				#print(all_loss_train)
				#pickle.dump(all_loss_train,open('/data/gabriel/train_val_loss_rnn/train_val_loss_rnn/new_train_hidden_dim_'+str(model_cl.layers)+'_loss.pkl','wb'))
				#pickle.dump(all_loss_val,open('/data/gabriel/train_val_loss_rnn/train_val_loss_rnn/new_val_hidden_dim_'+str(model_cl.layers)+'_loss.pkl','wb'))
			else:
				all_loss_val.append(running_loss_val)		
				#print(epoch_list)
				#print(all_loss_train)
				vid_acc = conf_m_vid.diagonal().sum()/conf_m_vid.sum()
				img_acc = conf_m_img.diagonal().sum()/conf_m_img.sum()
				print('video accuracy =',vid_acc)
				print('image accuracy =',img_acc)

				if(img_acc >= best_val_acc):
					best_val_acc = img_acc
					best_model = model_cl
					print('best_epoch = ',epoch)
				plt.plot(epoch_list,all_loss_val)


				plt.savefig('/data/gabriel/train_loss_rnn/val_bneck_pre_l'+str(model_cl.layers)+'_hd_'+str(model_cl.hidden_dim)+'_dr_'+str(model_cl.dropout)+'_loss.png')
				plt.close()


		random.shuffle(train_data_keys)
		random.shuffle(val_data_keys)
		data_keys = {'train':train_data_keys,'val':val_data_keys}
	#print(len(all_loss_train))
	#print(all_loss_train)
	#print(len(np.arange(0,model_cl.epochs)))
	#plt.plot(all_loss_train,np.arange(0,model_cl.epochs))
	#plt.savefig('/data/gabriel/train_loss_rnn/new_train_hidden_dim_'+str(model_cl.layers)+'_loss.png')
	#plt.close()
	#pickle.dump(all_loss_train,open('/data/gabriel/train_val_loss_rnn/train_val_loss_rnn/new_train_hidden_dim_'+str(model_cl.layers)+'_loss.pkl','wb'))
	#pickle.dump(all_loss_val,open('/data/gabriel/train_val_loss_rnn/train_val_loss_rnn/new_val_hidden_dim_'+str(model_cl.layers)+'_loss.pkl','wb'))
	
	#plt.plot(all_loss_val,np.arange(0,model_cl.epochs))
	#plt.savefig('/data/gabriel/train_loss_rnn/train_val_loss_rnn/val_hidden_dim_'+str(model_cl.layers)+'_loss.png')
	#plt.close()
	#print(model_cl.load_state_dict(best_model_wts))
	return model_cl,op,num_to_lab
import os


from scipy import stats
import pickle
def test(model_cl,res_dir):

	if(not os.path.exists(res_dir)):
		os.makedirs(res_dir)

	test_dic = {'test':model_cl.load_data('test')}
	test_data_keys = list(test_dic['test'].keys())
	model_cl.eval()
	model_cl = model_cl.cuda()
	
	img_acc = 0
	vid_acc = 0
	total_im = 0
	total_vid = 0


	conf_m_vid = np.zeros((model_cl.num_views,model_cl.num_views))
	conf_m_img = np.zeros_like(conf_m_vid)

	misc_cl = {}

	for keys in test_data_keys:
		#print(keys)
		embeds,label,label_name,name_file = test_dic['test'][keys]

		embeds = Variable(embeds.cuda())
		
		ac_label = label[0]

		label = torch.FloatTensor(embeds.size(0)).fill_(label[0])
		#label = Variable(label.cuda())
		#print(label)
		out = model_cl(embeds)
		#print(out)
		out_cpu = out.cpu()
		_,pred_lab = torch.max(out_cpu.data,1)
		#print(pred_lab)
		#print(pred_lab)
		for i in range(0,pred_lab.size(0)):
			# print(type(i))
			# print(type(label[i]))
			# print(type(pred_lab[i]))
			#print(type(pred_lab[i]))
			#print(pred_lab[i][0])
			conf_m_img[label.numpy().astype(int)[i],pred_lab[i]]+=1

		pred_vid = stats.mode(pred_lab.numpy(),axis=None)
		#print(pred_vid)
		#print(pred_lab)
		#print(ac_label)
		#print(pred_vid[0])
		conf_m_vid[ac_label,pred_vid[0]]+=1

		if(pred_vid[0]!=ac_label):
			# print(pred_vid[0])
			# print(ac_label)
			misc_cl[keys] = [model_cl.labels[pred_vid[0][0]],model_cl.labels[int(ac_label)]]

			### format is --- 'video_numframes -- > predicted label , actual label'

		#exit()
		#is_eq =	torch.eq(pred_lab,out_cpu.data).numpy()

		#img_acc+=torch.sum(torch.eq(label,out_cpu.data))
	m_name = res_dir[res_dir.find('s_lstm'):res_dir.find('.pth')]
	#print(res_dir)
	print(m_name)
	pickle.dump(conf_m_vid,open(res_dir+'/'+m_name+'_video_conf_m.pkl','wb'))
	pickle.dump(conf_m_img,open(res_dir+'/'+m_name+'_image_conf_m.pkl','wb'))
	pickle.dump(misc_cl,open(res_dir+'/'+m_name+'_misc_videos.pkl','wb'))

	print(conf_m_vid)
	
	print(conf_m_img)
	print(misc_cl)

	print('video accuracy = ',conf_m_vid.diagonal().sum()/conf_m_vid.sum())
	print('image accuracy = ',conf_m_img.diagonal().sum()/conf_m_img.sum())
