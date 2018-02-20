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


def rename_folder(aug_name,dest_dir,temp_dir,base_name):
	#### FUNCTION TO RENAME THE OUTPUT FOLDER OF THE AUGMENTED IMAGES 
	#### AS WELL AS THE IMAGES THEMSELVES 
	
	#print(dest_dir)
	if(not os.path.isdir(temp_dir)):
		os.makedirs(temp_dir)

	if(not os.path.isdir(temp_dir+'/'+base_name)):
		os.makedirs(temp_dir+'/'+base_name)

	
	new_dest_fold = 'output_' + str(aug_name) 
	#shutil.rmtree(dest_dir+'/0')
	new_dest_dir = temp_dir + '/'+base_name+'/' + new_dest_fold
	shutil.copytree(dest_dir,new_dest_dir)
	shutil.rmtree(dest_dir)
	#path1 = src_dir+'/'+class_name+'/output_b/'
	
	
	count=1

	for i in os.listdir(new_dest_dir):
		if(len(i)>2):
			new_name = i[:find_2(i)-1]+aug_name+'_'+str(count) +'.jpg'
			count+=1

			#print("new_name")
			#print(i[:find_2(i)])
			#print(new_name)
			os.rename(new_dest_dir+'/'+i,new_dest_dir+'/'+new_name)

	return new_dest_dir

def remove_folder(out_folder_dir,class_dir):
	for i in os.listdir(out_folder_dir):
		if(os.path.isfile(out_folder_dir+'/'+i)):
			shutil.copy2(out_folder_dir+'/'+i,class_dir)

	



def aug(data_dir,src_dir,temp_dir,aug_types = 'skew_h'):
	### Augments images in a <phase> directory and stores them separately
	### data_dir is the directory that starts with 'test' or 'val'

	try:
		shutil.rmtree(temp_dir)
		os.makedirs(temp_dir)
	except:
		os.makedirs(temp_dir)

	list_class_name = [i for i in os.listdir(data_dir) if os.path.isdir(data_dir+'/'+i)]
	if(not(isinstance(aug_types,list))):
		aug_types=list(aug_types)
		### Forcing it to a list to iterate through string inputs instead of list of string inputs
	
	# if(not(os.path.isdir(src_dir))):
	# 	os.makedirs(src_dir)

	for class_name in list_class_name:
		try:
			shutil.rmtree(src_dir+'/'+class_name)
			shutil.copytree(data_dir+'/'+class_name,src_dir+'/'+class_name)
		except:

			shutil.copytree(data_dir+'/'+class_name,src_dir+'/'+class_name)

		class_name_dir = src_dir+'/'+class_name+'/'
		### Folders are requires since augmentor names files based on the folder images are present in
		for fol_name in get_base_names(data_dir+'/'+class_name):
			### Make folder per video containing all the frames of that video indexed as Pnum_vidnum
			
			print(fol_name)

			try:
				shutil.rmtree(src_dir+'/'+class_name+'/'+fol_name)
				os.makedirs(src_dir+'/'+class_name+'/'+fol_name)
			except:
				os.makedirs(src_dir+'/'+class_name+'/'+fol_name)
			
			# This moves all the patient_video images to their folder.
			for img_name in os.listdir(class_name_dir):
				if(os.path.isfile(class_name_dir+'/'+img_name) and img_name[:find_2(img_name)+1] == fol_name):
					shutil.move(class_name_dir+'/'+img_name,src_dir+'/'+class_name+'/'+fol_name)
					#print('here')

			#fol_img = [i for i in os.listdir(data_dir+'/'+class_name) if fol_name in i]
			#print(len(os.listdir(class_name_dir+'/'+fol_name)))
			new_dest_dir_list = []
	
			for aug_type in aug_types:
				## Apply augmentations to all the images in that folder
				p = Augmentor.Pipeline(class_name_dir+'/'+fol_name)
				#print(class_name_dir)
				#print(fol_name)
				#print(aug_type)
				if(aug_type=='skewh'):
					p.skew_left_right(probability=1,magnitude = 0.5)
				elif(aug_type=='skewv'):
					p.skew_left_right(probability=1,magnitude = 0.5)
				elif(aug_type=='gauss'):
					p.gaussian_distortion(probability=1,grid_width = 8,grid_height=8,magnitude = 9,corner='bell',method='in')
				elif(aug_type=='rand'):
					p.random_distortion(probability=1,grid_width = 7,grid_height=7,magnitude = 9)

				p.sample(len(os.listdir(src_dir+'/'+class_name+'/'+fol_name))) ### This thus stores all the augmented images in fol_name/output/
				
				#shutil.rmtree(class_name_dir+'/'+fol_name+'/output/0')
				
				dest = src_dir+'/'+class_name+'/'+fol_name+'/output/'
				
				new_dest_dir_list.append(rename_folder(aug_name=aug_type,dest_dir=dest,temp_dir = temp_dir,base_name=fol_name))
				del(p)
			print(new_dest_dir_list)
				
			for new_dest in new_dest_dir_list:
				
				remove_folder(new_dest,class_name_dir)

			shutil.rmtree(temp_dir+'/'+fol_name)

		#Finally after moving everything, remove all folders in class_name_dir
		for i in os.listdir(class_name_dir):
			if(os.path.isdir(class_name_dir+'/'+i)):
				print(class_name_dir+'/'+i)
				shutil.rmtree(class_name_dir+'/'+i)

	shutil.rmtree(temp_dir)

#aug('/data/gabriel/VC_1/SET7/dataset/test2/','/data/gabriel/VC_1/SET7/dataset/test_distort23/','/data/gabriel/temp_dir/',['skewh','gauss'],)
