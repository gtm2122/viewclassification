import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import torch.optim as optim
import torch.nn.functional as F

import random

### This script is essentially the neural network architecture along with wrappers for training and testing

class lstm_proc(nn.Module):
	def __init__(self,window_len=0,data_dir=0,cache_dir=0,overwrite=0,epochs=10,hidden_dim = 1000,num_views=7,embed_sz=19872,layers=1,dropout = 0):
		super(lstm_proc,self).__init__()
		self.epochs=epochs
		self.window_len = window_len
		self.cache_dir = cache_dir
		self.hidden_dim=hidden_dim
		self.num_views=num_views
		self.embed_sz = embed_sz
		self.data_dir = data_dir
		
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
		out1,self.hidden = self.lstm(x.view(self.window_len,1,self.embed_sz),self.hidden)
		
		return F.log_softmax(self.cl(out1).view(len(x),-1))

		

	def init_hidden(self):
		return (Variable(torch.zeros(self.layers,1,self.hidden_dim)),Variable(torch.zeros(self.layers,1,self.hidden_dim)))
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
			max_f = self.window_len
			
			#print(max_f)
			data_dic = {}
			for cl in os.listdir(self.data_dir+'/'+phase+'/'):
				if '.p' not in phase and '.p' not in cl:
					for vid in os.listdir(self.data_dir+'/'+phase+'/'+cl):
						if '.p' not in vid:
							
							if(self.window_len == -1):

								max_f  = len(os.listdir(self.data_dir+'/'+phase+'/'+cl+'/'+vid))
								#print(max_f)
							img_tensor = torch.zeros(max_f,1,self.embed_sz)
							count= 0
							for frame in os.listdir(self.data_dir+'/'+phase+'/'+cl+'/'+vid):
								#print(frame)
								img = torch.load(self.data_dir+'/'+phase+'/'+cl+'/'+vid+'/'+frame)

								if(self.window_len==-1):
									img_tensor[count,0,:] = torch.from_numpy(img)
								else:
									img_tensor[count % max_f,0,:] = torch.from_numpy(img)						
								count+=1

								if(count>0 and not(count%max_f) and self.window_len>0):
									#print(vid)
									data_dic[vid+str(count)] = (img_tensor,self.label_to_ix(cl))
									img_tensor = torch.zeros(max_f,1,self.embed_sz)
			
							if(self.window_len==-1):
								data_dic[vid+str(count)] = (img_tensor,self.label_to_ix(cl))
								img_tensor = torch.zeros(max_f,1,self.embed_sz)
			
			torch.save(data_dic,self.cache_dir)
			return data_dic
			#img_tensor = torch.zeros(max_f,)
			
			#for i in os.listdir(self.data_dir):

def train_net(model_cl):

	all_data_dic = {'train':model_cl.load_data('train'),'val':model_cl.load_data('val')}
	
	
	
	train_data_keys = list(all_data_dic['train'].keys())
	val_data_keys = list(all_data_dic['val'].keys())

	random.shuffle(train_data_keys)
	random.shuffle(val_data_keys)

	data_keys = {'train':train_data_keys,'val':val_data_keys}
	criterion = nn.NLLLoss().cuda()
	model_cl = model_cl.cuda()
	optimizer = optim.SGD(model_cl.parameters(),lr=0.005)
	all_loss_train = []
	all_loss_val = []
	for epoch in range(0,model_cl.epochs):
		running_loss_train = 0
		running_loss_val = 0
		
		for phase in ['train','val']:
		#for phase in ['train','val']:
			for keys in data_keys[phase]:
				#print(keys)
				model_cl.zero_grad()
				model_cl.hidden = (model_cl.init_hidden()[0].cuda(),model_cl.init_hidden()[1].cuda())
				embeds,label = all_data_dic[phase][keys]
				
				label = torch.LongTensor(embeds.size(0)).fill_(label[0])
				
				embeds = Variable(embeds.cuda())
				label = Variable(label.cuda())

				out = model_cl(embeds)
				
				loss = criterion(out.view(out.size(0),-1),label)

				if(phase =='train'):
					loss.backward()
					running_loss_train+= loss.cpu().data.numpy()[0]
					optimizer.step()
				if(phase=='val'):
					running_loss_val+= loss.cpu().data.numpy()[0]
					#optimizer.step()

				del(out)
				del(loss)
				del(label)
				del(embeds)
		all_loss_train.append(running_loss_train)
		all_loss_val.append(running_loss_val)

		random.shuffle(train_data_keys)
		random.shuffle(val_data_keys)
		data_keys = {'train':train_data_keys,'val':val_data_keys}
	print(len(all_loss_train))
	print(all_loss_train)
	print(len(np.arange(0,model_cl.epochs)))
	plt.plot(all_loss_train,np.arange(0,model_cl.epochs))
	plt.savefig('/storage/train_val_loss_rnn/train_hidden_dim_'+str(model_cl.layers)+'_loss.png')
	plt.close()
	
	plt.plot(all_loss_val,np.arange(0,model_cl.epochs))
	plt.savefig('/storage/train_val_loss_rnn/val_hidden_dim_'+str(model_cl.layers)+'_loss.png')
	plt.close()
	return model_cl,optimizer
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


	conf_m_vid = np.zeros((7,7))
	conf_m_img = np.zeros_like(conf_m_vid)

	misc_cl = {}

	for keys in test_data_keys:
		#print(keys)
		embeds,label = test_dic['test'][keys]

		embeds = Variable(embeds.cuda())
		
		ac_label = label[0]

		label = torch.FloatTensor(embeds.size(0)).fill_(label[0])
		#label = Variable(label.cuda())
		#print(label)
		out = model_cl(embeds)
		out_cpu = out.cpu()
		_,pred_lab = torch.max(out_cpu.data,1)
		
		for i in range(0,pred_lab.size(0)):
			# print(type(i))
			# print(type(label[i]))
			# print(type(pred_lab[i]))

			conf_m_img[label.numpy().astype(int)[i],pred_lab[i]]+=1

		pred_vid = stats.mode(pred_lab.numpy(),axis=None)
		#print(pred_vid)
		#print(pred_lab)
		conf_m_vid[ac_label,pred_vid[0]]+=1

		if(pred_vid[0]!=ac_label):
			# print(pred_vid[0])
			# print(ac_label)
			misc_cl[keys] = [model_cl.labels[pred_vid[0][0]],model_cl.labels[int(ac_label)]]

			### format is --- 'video_numframes -- > predicted label , actual label'

		#exit()
		#is_eq =	torch.eq(pred_lab,out_cpu.data).numpy()

		#img_acc+=torch.sum(torch.eq(label,out_cpu.data))
	m_name = res_dir[res_dir.find('saved_model'):res_dir.find('.pth')+1]

	pickle.dump(conf_m_vid,open(res_dir+'/'+m_name+'_video_conf_m.pkl','wb'))
	pickle.dump(conf_m_img,open(res_dir+'/'+m_name+'_image_conf_m.pkl','wb'))
	pickle.dump(misc_cl,open(res_dir+'/'+m_name+'_misc_videos.pkl','wb'))

	print(conf_m_vid)
	print(conf_m_img)
	print(misc_cl)

