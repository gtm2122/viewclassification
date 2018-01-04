from test_model import *
#m_dir = '/data/Gabriel_done_deid/G4_imgs/0KXGJOAU'
#m_dir = '/data/gabriel/bottleneck_codes_echo_pre/test/PSA/'
m_dir1 = ['/data/Gabriel_done_deid/G4_imgs/'+i for i in os.listdir('/data/Gabriel_done_deid/G4_imgs/') if '.' not in i]
for m_dir in m_dir1:
	print(m_dir)
	m = get_extractor(model_dir='/data/gabriel/VC_1/SET7/DenseNetModel_pretrained_7_views_bs_64_e_50_26092017_182851.pth.tar')
	ls = torch.load('/data/gabriel/bottleneck_codes_echo_pre/saved_model_layers2.pth')
	predict(main_dir = m_dir,fe_model_obj=m,lstm_obj=ls)
