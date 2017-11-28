import sys
sys.path.append('../')

from nndev import model_pip
from bottleneck import gen_b
from torch import nn
from torchvision import models

setnum = sys.argv[1]
setnum = setnum[setnum.find('SET'):]
import os
print(setnum)
gpu = sys.argv[2]
print(gpu)
batch_size11 = sys.argv[3]
model_name = [i for i in os.listdir('/storage/VC_2/'+setnum) if '.pth.tar' in i][0]

res = models.densenet161(pretrained=True) 
res.classifier = nn.Linear(res.classifier.in_features,15)

obj1 = model_pip(model_in=res,data_path = '/storage/VC_2/'+setnum+'/dataset/',batch_size=1, gpu=0,scale = True,use_gpu=True,resume=False,verbose=False)
obj1.load_model('/storage/VC_2/'+setnum+'/'+model_name)
ab2 = gen_b(model1 = obj1.model ,data_dir='/storage/VC_2/'+setnum+'/dataset/',save_dir='/storage/'+setnum+'_bnecks/',b_size= batch_size11, gpu_num=gpu) 
ab2.get_f()

with open(ab2.save_dir+setnum+'.txt','w') as ff:
	ff.write('DONE')
