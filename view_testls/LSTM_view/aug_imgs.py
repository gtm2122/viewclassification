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
			list_names.append(i[:find_2(i)+1])

	return list(set(list_names))


def aug(data_dir,src_dir,aug_type = 'skew'):
	### Augments images in a <phase> directory and stores them separately
	### data_dir is the directory that starts with 'test' or 'val'

	list_class_name = [i for i in os.listdir(data_dir) if os.path.isdir(data_dir+'/'+i)]

	for class_name in list_class_name:
		try:
			shutil.rmtree(src_dir+'/'+class_name)
			shutil.copytree(data_dir+'/'+class_name,src_dir+'/'+class_name)
		except:
<<<<<<< HEAD
			shutil.copytree(data_dir+'/'+class_name,src_dir+'/'+class_name)

		p = Augmentor.Pipeline(src_dir+'/'+class_name+'/')

		if(aug_type=='skew'):
			p.skew_left_right(probability=1,magnitude = mag)
		else:
			p.gaussian_distortion(probability=1,grid_width = 8,grid_height=8,magnitude = mag,corner='bell',method='in')

		p.sample(len(os.listdir(src_dir+'/'+class_name+'/'))-1)

		os.rename(src_dir+'/'+class_name+'/output',src_dir+'/'+class_name+'/output_a')
		path1 = src_dir+'/'+class_name+'/output_a/'
		
		all_base_names = get_base_names(data_dir)
		unique_dic = {x:1 for x in all_base_names}
		shutil.rmtree(src_dir+'/'+class_name+'/output_a/0')

		for i in os.listdir(path1):
			print(path1)
			print(i)
			new_name = i[:find_2(i)]+'a_'+str(unique_dic[i[:find_2(i)+1]] +'.jpg')
			unique_dic[i[:find_2(i)+1]] +=1
			os.rename(path1+'/'+i,path1+'/'+new_name)

		del(p)
		p = Augmentor.Pipeline(src_dir+'/'+class_name+'/')
		
		if(aug_type=='skew'):
			p.skew_top_bottom(probability=1,magnitude = mag)
		else:
			p.random_distortion(probability=1,grid_width = 7,grid_height=7,magnitude = mag)
		

		p.sample(len(os.listdir(src_dir+'/'+class_name+'/'))-1)
		os.rename(src_dir+'/'+class_name+'/output',src_dir+'/'+class_name+'/output_b')
		path1 = src_dir+'/'+class_name+'/output_b/'
		shutil.rmtree(src_dir+'/'+class_name+'/output_b/0')
		all_base_names = get_base_names(data_dir)
		unique_dic = {x:1 for x in all_base_names}
		
		for i in os.listdir(path1):
			new_name = i[:find_2(i)]+'b_'+str(unique_dic[i[:find_2(i)+1]] +'.jpg')
			unique_dic[i[:find_2(i)+1]] +=1
			os.rename(path1+'/'+i,path1+'/'+new_name)



		del(p)
		
		for new_img_path in os.listdir(src_dir+'/'+class_name+'/'+fol_name+'/output_a/'):
			path_img = src_dir+'/'+class_name+'/output_a/'+new_img_path
			temp_p = src_dir+'/'+class_name+'/output_a/'

			for f_name in os.listdir(src_dir+'/'+class_name+'/output_a/'):
				
				shutil.copy2(temp_p+f_name,src_dir+'/'+class_name+'/')

		for new_img_path in os.listdir(src_dir+'/'+class_name+'/'+fol_name+'/output_b/'):
			path_img = src_dir+'/'+class_name+'/output_b/'+new_img_path
			temp_p = src_dir+'/'+class_name+'/output_b/'

			for f_name in os.listdir(src_dir+'/'+class_name+'/output_b/'):
				
				shutil.copy2(temp_p+f_name,src_dir+'/'+class_name+'/')

=======
			os.makedirs(src_dir+'/'+class_name)

		for fol_name in get_base_names(data_dir+'/'+class_name)[:10]:
			
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

			if(aug_type=='skew'):
				p.skew_left_right(probability=1,magnitude = 0.5)
			else:
				p.gaussian_distortion(probability=1,grid_width = 8,grid_height=8,magnitude = 9,corner='bell',method='in')

			p.sample(len(os.listdir(src_dir+'/'+class_name+'/'+fol_name)))

			del(p)
			p = Augmentor.Pipeline(src_dir+'/'+class_name+'/'+fol_name)
			
			if(aug_type=='skew'):
				p.skew_top_bottom(probability=1,magnitude = 0.5)
			else:
				p.random_distortion(probability=1,grid_width = 7,grid_height=7,magnitude = 9)
>>>>>>> parent of 1a1298d... added magnitude
		
		shutil.rmtree(src_dir+'/'+class_name+'/'+'/output_b/')
		shutil.rmtree(src_dir+'/'+class_name+'/'+'/output_a/')

#aug('/data/gabriel/VC_1/SET7/dataset/test/','/data/gabriel/VC_1/SET7/dataset/test_distort2/')
