from test_model import *
#m_dir = '/data/Gabriel_done_deid/G4_imgs/0KXGJOAU'
#m_dir = '/data/gabriel/bottleneck_codes_echo_pre/test/PSA/'
os.environ['CUDA_VISIBLE_DEVICES']='1'
model_names = [i for i in os.listdir('/data/gabriel/viewclassification/view_testls/LSTM_view/saved_lstm2/') if 'batch' in i and '.pkl' not in i ]
#model_names = '/data/gabriel/saved_lstm//lstm_hd1500_layers_2_dr_0'#] ## Top 3
main_path = '/data/gabriel/viewclassification/view_testls/LSTM_view/saved_lstm2/'
model_obj_list = [torch.load(main_path+i).eval().cuda() for i in model_names]

fe = get_extractor('/data/gabriel/VC_1/SET7/DenseNetModel_pretrained_7_views_bs_64_e_50_26092017_182851.pth.tar').eval().cuda()

with open('errorlog.txt','w') as fi:
	fi.write('___________________________________ERROR LOG________________________________')

#output_tr = {0:'PSA',5:'PLA',6:'A3C',3:'A4C',4:'A2C',1:'SCX',2:'SSX'}
output_tr = pickle.load(open('/data/gabriel/viewclassification/view_testls/LSTM_view/saved_lstm2/NEW_KEY_NUM_TO_LAB.pkl','rb'))
result = []
for sub_f in os.listdir('/data/Gabriel_done_deid/G4_imgs/'):

	result+=(predict(main_dir='/data/Gabriel_done_deid/G4_imgs/'+sub_f,fe_model_obj=fe,
		lstm_obj=model_obj_list,output_legend=output_tr,model_name='densenet',num_views=15,batch_size=3,window_len=5,k_size=9))	

print(result)
with open('./G4_15_view_batch_lstm_results_test.csv','w') as f:
	writer = csv.writer(f)
	writer.writerows(result)

del(fe)
del(model_obj_list)