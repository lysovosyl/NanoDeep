import csv
import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from read_deep.rt import rt_deep
from importlib import import_module
import yaml


#%%
def main():
    parser = argparse.ArgumentParser(description="you should add those parameter")
    parser.add_argument('-data_path', required=True, type=str, help='The data used to train model')
    parser.add_argument('-label_path', required=True, type=str, help='The label used to train model')
    parser.add_argument('-save_path', required=True, type=str, help='The dir used to save train result')
    parser.add_argument("-model_path",required=True,type=str,help='The model which used to test')
    parser.add_argument('-model_name', default='Nanodeep', help='The model to be used')
    parser.add_argument('-batch_size',default=50,type=int,help='The amount of data used to train the model once')
    parser.add_argument('-device', default='cuda:0', help='The device use to train model')
    parser.add_argument('-model_config', default=None, help='Model hyperparameter settings')
    parser.add_argument('-signal_length', default=5000, type=int,
                        help='How long signals are used for classification, note that the first 1500 signals are removed')
    parser.add_argument('--load_to_mem', default=False, action='store_true',
                        help='All train data will be load to memory,it will be faster')
    opt = parser.parse_args()

    print('-model_name:', opt.model_name)
    print('-model_path:', opt.model_path)
    print('-data_path:', opt.data_path)
    print('-label_path:', opt.label_path)
    print('-save_path:', opt.save_path)
    print('-batch_size:', opt.batch_size)
    print('-device:', opt.device)
    print('-model_config:', opt.model_config)
    print('-signal_length:', opt.signal_length)
    print('-load_data_to_mem:', opt.load_to_mem)

    if opt.model_config == None:
        model_args = getattr(import_module('read_deep.model_config.defaultconfig'), opt.model_name)
    else:
        ymlfile = open(opt.model_config,'r',encoding='utf-8')
        model_args = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    experiment_save_path = os.path.join(opt.save_path)
    if not os.path.exists(experiment_save_path):
        os.makedirs(experiment_save_path)
    roc_save_path = os.path.join(experiment_save_path)
    csv_save_path = os.path.join(experiment_save_path, 'acc.csv')

    deepmodel = getattr(import_module('read_deep.model.'+opt.model_name),'model')
    torch.cuda.set_device(opt.device)
    model = deepmodel(**model_args)
    nanopore_gpu = rt_deep(model, opt.signal_length,opt.device)
    nanopore_gpu.load_the_model_weights(opt.model_path)
    print("load weight complete")
    nanopore_gpu.load_data(data_path=opt.data_path,
                           label_path=opt.label_path,
                           dataset='test',
                           load_to_mem=opt.load_to_mem)
    confmat_data, acc_value, lable_all, predict_proba = nanopore_gpu.test_model(batch_size = opt.batch_size)
    nanopore_gpu.draw_ROC(lable_all, predict_proba, roc_save_path)
    f = open(csv_save_path,'w')
    writer = csv.writer(f)
    writer.writerows(confmat_data)
    writer.writerow(['accuracy:'+str(acc_value)])
    f.close()

if __name__ == '__main__':
    main()



