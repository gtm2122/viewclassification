import matplotlib.pyplot as plt

import  torch.utils.data as data_utils
import torch
import torch.nn as nn
from torchvision.models import inception
from torchvision.models import Inception3
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets,models,transforms
import torch.optim as optim
import copy

import skimage.io as io
from scipy.misc import imsave
from skimage import img_as_uint
import errno 
import numpy as np
import random
import gc


import torch
import numpy as np
from torch.autograd import Variable

import sys
import os
import scipy
import cv2
from PIL import Image
class model_pip(object):

    def __init__(self,model_in,data_path = '/data/gabriel/OCR/OCR_data/',
                 model_saved = '/data/gabriel/VC/model/1.pth.tar',
                 batch_size=90,lr = 0.001,epochs=10, 
                 gpu=0,f_extractor = False,scale = False,op=optim.SGD,
                 criterion=nn.CrossEntropyLoss(),use_gpu=True,lr_decay_epoch=7,resume=False):
        self.epochs=int(epochs)
        self.lr_decay_epoch=lr_decay_epoch
        self.data_path = data_path
        self.model = model_in
        self.b_size = batch_size
        self.scale = scale
        self.lr = lr
        self.criterion = criterion
        self.fe = f_extractor
        self.gpu=gpu
        self.use_gpu=use_gpu
        self.model_optimizer = op(model_in.parameters(),lr = self.lr,momentum=0.9)
        self.model_saved = model_saved
        self.resume = resume
        self.num_output = len(os.listdir(data_path+'train/'))
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
    def transform(self,rand = False):
        if(self.scale==True):
            
            
            if(rand):
                data_transforms = {'train':transforms.Compose([transforms.Scale(300),
                                            transforms.CenterCrop(300),
                                                   #transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                       ,
                       'val':transforms.Compose([transforms.Scale(350),
                transforms.RandomCrop(300),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                ])
                }
            
            else:
                
                data_transforms = {'train':transforms.Compose([transforms.Scale(300),
                                            transforms.CenterCrop(300),
                                               #transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                       ,
                       'val':transforms.Compose([transforms.Scale(300),
                transforms.CenterCrop(300),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                ])
                }
        else:
            data_transforms = {'train':transforms.Compose([transforms.CenterCrop((646,434)),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                   ,
                   'val':transforms.Compose([
            transforms.CenterCrop((646,434)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
            }
        dsets = {x:datasets.ImageFolder(os.path.join(self.data_path,x),data_transforms[x])
            for x in ['train','val']}

        dset_loaders ={x: torch.utils.data.DataLoader(dsets[x],batch_size=self.b_size,shuffle=True,num_workers=4)
              for x in ['train','val']}

        dset_sizes = {x:len(dsets[x]) for x in ['train','val']}
        
        return dsets,dset_loaders,dset_sizes
    
    #@staticmethod
    def lr_scheduler(self,epoch):
        lr = self.lr*(0.1**(epoch//self.lr_decay_epoch))
        if epoch%self.lr_decay_epoch ==0:
            print('lr = ',lr)
        for param_group in self.model_optimizer.param_groups:
            param_group['lr'] = lr
        #return optimz
    
    #@staticmethod
    def imshow(inp,title=None):
        inp=inp.numpy().transpose((1,2,0))
        mean=np.array([0.485,0.456,0.406])
        std = np.array([0.229,0.224,0.225])
        inp = std*inp + mean
        plt.imshow(inp),plt.show()
        plt.pause(0.01)
    
    
  
    def load_model(self,filename):
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epochs']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model_optimizer.load_state_dict(checkpoint['optimizer'])
        #return model,optimz,start_epoch
    
    def train_model(self, n=None):
        dsets,dset_loaders,dset_sizes = self.transform() 
        print(dset_sizes)
        
        model = self.model
        #op = self.model_optimizer
        epoch_init = 0
        
        '''
        if(self.resume):
            
            try:
                self.load_model()
            except:
                print('invalid directory, starting from scratch')
        '''
        
        best_model=model
        
        criterion = self.criterion
        
        if(torch.cuda.is_available() and self.use_gpu):
            torch.cuda.set_device(self.gpu)
            model=model.cuda()
            criterion=criterion.cuda()
        
        best_acc = 0.0
        best_epoch = 0
        
        
        if(~self.fe):
        
            for epoch in range(epoch_init,self.epochs):
                print('Epoch = ',epoch)

                for phase in ['train','val']:
                    if(phase == 'train'):
                        model.train(True)

                        self.lr_scheduler(epoch)
                    else:
                        model.train(False)
                    c_mat = np.zeros((self.num_output,self.num_output)) 
                    running_loss = 0.0
                    running_corrects = 0.0
                    running_tp = 0.0
                    for data in dset_loaders[phase]:
                        inputs,labels = data
                        #print(inputs.size())
                        if(torch.cuda.is_available() and self.use_gpu):
                            inputs,labels = Variable(inputs.cuda()),Variable(labels.cuda())
                        else:
                            inputs,labels = Variable(inputs),Variable(labels)
                        
                        self.model_optimizer.zero_grad()
                        flag=0
                        
                        if(inputs.size(0)<self.b_size and  n == 'Inception'):
                            flag=1
                            
                            if(torch.cuda.is_available() and self.use_gpu):   
                                temp = Variable(torch.zeros((self.b_size,3,300,300)).cuda())
                                temp2 = Variable(torch.LongTensor(self.b_size).cuda())
                                             
                            else:
                                temp=Variable(torch.zeros((self.b_size,3,300,300)))
                                temp2=Variable(torch.LongTensor(self.b_size))
                                             
                            temp[0:inputs.size(0)]=inputs
                            inputs = temp
                            
                            temp2[0:labels.size(0)] = labels
                            temp2[labels.size(0):] = 0
                            labels = temp2
                        
                        outputs = model(inputs)
                        #print(outputs.size())
                        #print(labels.size())
                        if(n=='Inception'):
                            if phase=='val':

                                _,preds = torch.max(outputs.data,1)
                                loss = criterion(outputs,labels)
                            else:
                                _,preds = torch.max(outputs[0].data,1)
                                loss = criterion(outputs[0],labels)

                        else:
                            _,preds = torch.max(outputs.data,1)
                            loss = criterion(outputs,labels)



                        if phase=='train':
                            loss.backward()
                            self.model_optimizer.step()
                        
                        running_loss+=loss.data[0]
                        running_corrects += torch.sum(preds == labels.data)
                        for i in range(0,labels.data.cpu().numpy().shape[0]):

                            c_mat[labels.data.cpu().numpy()[i],preds.cpu().numpy()[i]]+=1


                    epoch_loss = running_loss/dset_sizes[phase]
                    epoch_acc = running_corrects/dset_sizes[phase]
                    #epoch_tpr = running_tp/dset_sizes[phase]
                    
                    print(phase + '{} Loss: {:.4f} Acc: {:.4f}'.format(phase,epoch_loss,epoch_acc))
                    #print(c_mat)
                    print(c_mat)
                    if phase=='val' and epoch_acc>best_acc:
                        best_acc=epoch_acc
                        best_model=copy.deepcopy(model)
                        best_epoch=epoch

                #print()

        print(best_acc)
        print(best_epoch)
            
        self.model=best_model.cpu()
        
    def store_model(self,f_name):
        
        
        state={'epochs':self.epochs,'state_dict':self.model.state_dict(),
                          'optimizer':self.model_optimizer.state_dict()}
        torch.save(state,f_name)
    
    def test(self,model_dir,n):
        #TODO modify class to add a test path
        self.model.eval()
        
        dsets,dset_loaders,dset_sizes = self.transform(rand=True)
        flag=False
        if(torch.cuda.is_available() and self.use_gpu):
            flag=True
        if(flag):
            torch.cuda.set_device(self.gpu)
            self.model.cuda()
        running_corrects = 0    
        for data in dset_loaders['val']:
            inp,labels = data
            c_mat = np.zeros((self.num_output,self.num_output)) 
            if(flag):
                inp,labels=inp.cuda(),labels.cuda()
        
            inp,labels=Variable(inp),Variable(labels)
            
            if(inp.size(0)<self.b_size and  n == 'Inception'):
                #flag=1
                            
                if(flag):   
                    temp = Variable(torch.zeros((self.b_size,3,300,300)).cuda())
                    temp2 = Variable(torch.LongTensor(self.b_size).cuda())
                                             
                else:
                    temp=Variable(torch.zeros((self.b_size,3,300,300)))
                    temp2=Variable(torch.LongTensor(self.b_size))
                                             
                temp[0:inputs.size(0)]=inp
                inp = temp
                            
                temp2[0:labels.size(0)] = labels
                temp2[labels.size(0):] = 0
                labels = temp2
                        

            
            output = self.model(inp)
            if(n=='Inception'):
                

                _,preds = torch.max(output.data,1)
                
            else:
                _,preds = torch.max(output.data,1)
                
            running_corrects += torch.sum(preds == labels.data)
            for i in range(0,labels.data.cpu().numpy().shape[0]):

                c_mat[labels.data.cpu().numpy()[i],preds.cpu().numpy()[i]]+=1

            epoch_acc = running_corrects/dset_sizes['val']
        print(c_mat)
        print(epoch_acc)
            
        np.savetxt(model_dir+'/c_mat.txt',c_mat.astype(int))
        np.savetxt(model_dir+'/accuracy.txt',np.array(epoch_acc).reshape(1,))
            
            