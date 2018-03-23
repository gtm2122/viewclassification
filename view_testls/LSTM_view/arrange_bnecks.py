import os
import shutil

#### Script for arranging each frame into it's video folder

def find_2(s):
	return [i for i,j in enumerate(s) if j=='_'][-1]

def arrange_data(d):
	unique=[]
	dirs = os.listdir(d)
	for i in dirs:
		#try:
		#print(i)
		if(os.path.exists(d+'/'+i)):
			#print('dere')
			if i[:find_2(i)] not in unique and '.jpg.pth' in i or '.JPEG.pth' in i:
				unique.append(i[:find_2(i)+1])
				if(not(os.path.exists(d+'/'+unique[-1]))):
					os.makedirs(d+'/'+unique[-1])
					print(d+'/'+unique[-1])
				for k,j in enumerate(dirs):
					if unique[-1] in j[:len(unique[-1])]:
						print(d+'/'+j,d+'/'+unique[-1]+'/')
						if('.' in d+'/'+j):
							shutil.copy(d+'/'+j,d+'/'+unique[-1]+'/')
		# except:
		# 	continue

def arrange_data_img(d):
	unique=[]
	dirs = os.listdir(d)
	for i in dirs:
		#try:
		#print(i)
		if(os.path.exists(d+'/'+i)):
			#print('dere')
			#print(d+'/'+i)
			if i[:find_2(i)] not in unique and '.jpg' in i or '.png' in i:
				unique.append(i[:find_2(i)+1])
				if(not(os.path.exists(d+'/'+unique[-1]))):
					os.makedirs(d+'/'+unique[-1])
					print(d+'/'+unique[-1])
				for k,j in enumerate(dirs):
					if unique[-1] in j[:len(unique[-1])]:
						print(d+'/'+j,d+'/'+unique[-1]+'/')
						if('.' in d+'/'+j):
							shutil.copy(d+'/'+j,d+'/'+unique[-1]+'/')
		# except:
		# 	continue


#for phase in ['train','test','val']:

"""
for phase in ['test_distort_2_16','train_distort_2_16','val_distort_2_16']:
	for c in os.listdir('/data/gabriel/SET1_bnecks_2_16/'+phase):
		if('.' not in c):
			arrange_data('/data/gabriel/SET1_bnecks_2_16/'+phase+'/'+c)
			print('/data/gabriel/SET1_bnecks_2_16/'+phase+'/'+c)

"""
for phase in ['train','val','test']:
	for c in os.listdir('/data/gabriel/dataset/'+phase):
		if('.' not in c):
			arrange_data_img('/data/gabriel/dataset/'+phase+'/'+c)
			print('/data/gabriel/dataset/'+phase+'/'+c)
