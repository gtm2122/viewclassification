import Augmentor
import os
import shutil



def find_2(s):

	return [i for i,j in enumerate(s) if j=='_'][-1]

def get_base_names(data_dir):
	list_names =[]
	for i in os.listdir(data_dir):
		#print(i)
		if(os.path.isfile(data_dir+'/'+i)):
			list_names.append(i[:find_2(i)])

	return list(set(list_names))


def aug(data_dir,src_dir):
	### Augments images in a <phase> directory and stores them separately
	### data_dir is the directory that starts with 'test' or 'val'

	list_class_name = [i for i in os.listdir(data_dir) if os.path.isdir(data_dir+'/'+i)]

	for class_name in list_class_name:
		try:
			shutil.rmtree(src_dir+'/'+class_name)
			os.makedirs(src_dir+'/'+class_name)
		except:
			os.makedirs(src_dir+'/'+class_name)

		for fol_name in get_base_names(data_dir+'/'+class_name):
			

			try:
				shutil.rmtree(src_dir+'/'+class_name+'/'+fol_name)
				os.makedirs(src_dir+'/'+class_name+'/'+fol_name)
			except:
				os.makedirs(src_dir+'/'+class_name+'/'+fol_name)
			
			fol_img = [i for i in os.listdir(data_dir+'/'+class_name) if fol_name in i]
			
			for j in fol_img:
				print(j)
				shutil.copy(data_dir+'/'+class_name+'/'+j,src_dir+'/'+class_name+'/'+fol_name)			

			p = Augmentor.Pipeline(src_dir+'/'+class_name+'/'+fol_name)

			p.gaussian_distortion(probability=1,grid_width = 8,grid_height=8,magnitude = 9,corner='bell',method='in')
			
			p.sample(len(os.listdir(src_dir+'/'+class_name+'/'+fol_name)))

			del(p)
			p = Augmentor.Pipeline(src_dir+'/'+class_name+'/'+fol_name)
			
			p.random_distortion(probability=1,grid_width = 8,grid_height=8,magnitude = 9)
			
			p.sample(len(os.listdir(src_dir+'/'+class_name+'/'+fol_name))-1)


aug('/data/gabriel/VC_1/SET7/dataset/test/','/data/gabriel/VC_1/SET7/dataset/test_distort/')