import torch
import os
import numpy as np

dif_sz = []

def size_check():
	for i in os.listdir('/data/gabriel/bottleneck_codes_echo_pre/'):
		for j in os.listdir('/data/gabriel/bottleneck_codes_echo_pre/'+i+'/'):
			if('.pkl' not in j):
				for k in os.listdir('/data/gabriel/bottleneck_codes_echo_pre/'+i+'/'+j):
					if('loader' not in k):
						for l in os.listdir('/data/gabriel/bottleneck_codes_echo_pre/'+i+'/'+j+'/'+k):
							if('.jpg.pth') in l:
								img = torch.load('/data/gabriel/bottleneck_codes_echo_pre/'+i+'/'+j+'/'+k+'/'+l)
								if img.shape not in dif_sz:
									dif_sz.append(img.shape)
	#print(dif_sz)
	print(dif_sz)

def GCD(a,b):
	if(b==0):
		return a
	else:
		return GCD(b,a%b)
from functools import reduce
def largest_folder():
	uniq_len = {}
	for i in os.listdir('/data/gabriel/bottleneck_codes_echo_pre/'):
		for j in os.listdir('/data/gabriel/bottleneck_codes_echo_pre/'+i):
			if('.pkl' not in j):
				for k in os.listdir('/data/gabriel/bottleneck_codes_echo_pre/'+i+'/'+j):
					#print(k)
					if('.p' not in k):
						num =len(os.listdir('/data/gabriel/bottleneck_codes_echo_pre/'+'/'+i+'/'+j+'/'+k))

						uniq_len[i+'/'+j+'/'+k] = num
					
	def loc_fun(k,dic = uniq_len):
		return dic[k]
	sorted_keys = sorted(uniq_len,key=loc_fun,reverse=True)
	#print(uniq_len.keys())
	print(reduce(lambda x,y : GCD(x,y), [uniq_len[i] for i in uniq_len]) )

	# for i in range(0,len(sorted_keys)):
	# 	print(sorted_keys[i],uniq_len[sorted_keys[i]])

largest_folder()