import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from lstm_class import find_2
import pickle
import re
import random

lab_to_ix = {x:i for i,x in enumerate(os.listdir('/data/gabriel/SET1_bnecks_normed/test/'))}

def load_data(data_dir,phase,dir_save_path = '/home/gam2018/cached_dsets/img_paths/100_pat_dset',step_size=1,ow=False,truncate=True):
    assert step_size>0 and isinstance(step_size,int),"step size must be a whole number"
    ###
    # data_dir = the dataset containing "test","train" etc. The contents of the folder are video folers containing
    # the CNN features extracted for each frame
    # phase = "test"/"train"/"val"
    # ow = overwrite list of paths ?
    # Truncate = Should the batches be limited to having whole number multiple of batch size or not? If not then we must modify the
    # loss function to not include the tail end zeros.
    class_list = os.listdir(data_dir+'/'+phase)
    seq_len = 10# number of images in a sequence
    batch_size = 100 # number of sequences in a batch
    embed_sz = 2208
    if not (os.path.isfile(dir_save_path+'/'+str(seq_len)+'_'+phase+'.pkl')) or ow:
        print("generating new list of image paths")
        img_path_list = []


        for label in class_list:
            for vid_folder in os.listdir(data_dir+'/'+phase+'/'+label):
                vid_path = data_dir+'/'+phase+'/'+label+'/'+vid_folder
                num_frames = len(os.listdir(vid_path))
                rem_imgs = num_frames%seq_len
                c = 0
                #list_of_imgs = os.listdir(vid_path)
                list_of_imgs = sorted(os.listdir(vid_path),key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)])
                #print(list_of_imgs)
                #exit()
                while c <= num_frames-seq_len :
                    seq_imgs_names = list_of_imgs[c:c+seq_len]
                    seq_imgs_names = [vid_path+'/'+i for i in seq_imgs_names]
                    img_path_list.append((seq_imgs_names,label,c))
                    c+=1 ## using step size 1 to get all possible windows
                    ### The above list will contain all possible windows in a list
                    ### Storing 'c' to enable step size filtering - for eg if c%step_size==0 then include in batch.

        print(img_path_list[-1])

        pickle.dump(img_path_list,open(dir_save_path+'/'+str(seq_len)+'_'+phase+'.pkl','wb'))
    else:
        img_path_list = pickle.load(open(dir_save_path+'/'+str(seq_len)+'_'+phase+'.pkl','rb'))
        if (step_size>1):
            new_path_list = [i for i in img_path_list if i[2]%step_size==0]
        else:
            new_path_list = img_path_list
        del img_path_list
        total_num_seq = len(new_path_list)
        ### Now that we loaded all the img paths for each sequence, we arrange various sequences into a minibatch
        if(truncate):
            print(len(new_path_list))
            total_num_batches = (total_num_seq // batch_size)
            new_path_list = new_path_list[:total_num_batches*batch_size]
            print(len(new_path_list))

        left_ind= 0

        right_ind = min(total_num_seq,left_ind + batch_size)

        random.shuffle(new_path_list)
        while left_ind < len(new_path_list):
            ### Two ways to deal with the last k sequences, one, truncate to multiple of batch_size, two, zero pad.
            ### Trying zero pad.
            print(left_ind)
            print(right_ind)
            mini_batch_paths = new_path_list[left_ind:right_ind]
            ### loop to load the images into tensors
            img_minibatch = torch.zeros((seq_len, batch_size, embed_sz))
            lab_minibatch = torch.zeros((batch_size))

            mini_batch_idx = 0
            for sequence_tuple in mini_batch_paths:
                ### Going through the 100 sequences
                #print(sequence_tuple[0][0])
                frame_num = 0
                for img_paths in sequence_tuple[0]:
                    ### Now going through the paths of the images in a sequence to fill in the frames of a sequence

                    img_minibatch[frame_num,mini_batch_idx,:] = torch.from_numpy(torch.load(img_paths))
                    frame_num+=1
                lab_minibatch[mini_batch_idx] = lab_to_ix[sequence_tuple[1]]

                mini_batch_idx+=1

            left_ind+=batch_size
            right_ind=min(left_ind+batch_size,len(new_path_list))

            ### if Truncate set to false then we should use the indices until right_ind for the loss function
            ### TODO Set option for truncate = False, return len(list)%batch_size to be used for loss function averaging.
            yield img_minibatch,lab_minibatch


for a,b in load_data('/data/gabriel/SET1_bnecks_normed/','test',step_size=5,truncate=True):
    #print(a)
    #print(b)
    1==1