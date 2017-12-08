import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import torch.optim as optim
import torch.nn.functional as F

import random

import os
from lstm_class import *
# a = torch.load('/data/gabriel/bottleneck_codes_echo_pre/test/A2C/88_20_/88_20_17.jpg.pth')
ddir = '/storage/SET2_bnecks/'

used = []

### varying Hidden dim and layers
for hd in [10,100,1000,2000]:
	for l in [2,3,4,5]:
		try:
			used.append(hd)
			bb = lstm_proc(num_views=15,data_dir = '/storage/SET2_bnecks/',cache_dir = ddir+'/cache.pth',overwrite=True,window_len = -1,hidden_dim = hd,epochs = 1000,layers=l)
			#print('1')
			#print('training')
			if(hd > 10 or l > 2):
				m,opt_m = train_net(bb)
				torch.save(bb,'/storage/saved_lstm/s_lstm_hd_'+str(hd)+'_layers_'+str(l)+'.pth')
			m = torch.load('/storage/saved_lstm/s_lstm_hd_'+str(hd)+'_layers_'+str(l)+'.pth')
			test(m,'/storage/saved_lstm/s_lstm_hd_'+str(hd)+'_layers_'+str(l)+'.pth')
		except:
			print('error at '+str(hd)+'_'+str(l))
			break
"""
hd = 500
ddir = '/storage/SET2_bnecks/'
for l in [2,3,4,5]:
	#used.append(hd)
	#bb = lstm_proc(data_dir = '/storage/SET2_bnecks/',cache_dir = ddir+'/cache.pth',overwrite=False,window_len = -1,hidden_dim = hd,epochs = 1000,layers = l)
	#print('1')
	#m,opt_m = train_net(bb)
	#torch.save(bb,'/storage/saved_lstm/saved_model_layers'+str(l)+'.pth')

	#m = torch.load('/storage/saved_lstm/saved_model_layers'+str(l)+'.pth')

	test(m,'/storage/saved_lstm/results_saved_model_layers'+str(l)+'.pth')

"""
