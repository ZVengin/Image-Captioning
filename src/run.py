from vocabulary import Vocabulary
from  train import main
opt={
    'data_dir': '../data_dir',
    'exp_dir': '../data_dir/exp_12',
    'raw_data_dir': '../raw_data_dir/annotations',
    'batch_size': 32,
    'num_workers':1,
    'shuffle':True,
    'encoder_name':'',
    'decoder_name':'',
    'pretrained': True,
    'num_epochs':135,
    'crop_size':224,
    'word_dime':512,
    'hidd_dime':512,
    'hidd_layer':1,
    'max_seq_leng':20,
    'lr':0.001,
    'log_step':2,
    'valid_step':10000,

}
main(opt)
