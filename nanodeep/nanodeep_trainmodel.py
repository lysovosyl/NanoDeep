import csv
import os
import argparse
from operator import add
from functools import reduce
from copy import deepcopy
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from read_deep.rt import rt_deep
from importlib import import_module
import yaml

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def k_fold_lable(label_path,k,save_path,assign_testdata:bool=False):
    file_name = os.listdir(label_path)
    lable = {}
    for i in file_name:
        file = open(os.path.join(label_path,i))
        lable[i] = []
        lines = file.readlines()
        for line in lines:
            id = line
            lable[i].append(id)
    split_lable = {}
    for i in file_name:
        split_lable[i] = []
        childern_list_len = int(len(lable[i])/k)+1
        for slice_index in range(0, len(lable[i]), childern_list_len):
            split_lable[i].append(lable[i][slice_index:slice_index + childern_list_len])

    label_path = {}
    label_path['test'] = []
    label_path['train'] = []
    label_path['validation'] = []
    for i in range(0,k):
        temp_lable = deepcopy(split_lable)
        for j in file_name:
            validation_or_test = temp_lable[j][i]
            temp_lable[j].pop(i)
            trainlable = temp_lable[j]
            trainlable = reduce(add, trainlable)

            lable_save_path = os.path.join(save_path, str(i))
            if os.path.exists(lable_save_path) == False:
                os.makedirs(lable_save_path)
            train_lable_save_path = os.path.join(lable_save_path, 'train')
            validation_lable_save_path = os.path.join(lable_save_path, 'validation')
            if os.path.exists(train_lable_save_path) == False:
                os.makedirs(train_lable_save_path)
            if os.path.exists(validation_lable_save_path) == False:
                os.makedirs(validation_lable_save_path)

            f = open(os.path.join(train_lable_save_path, j), 'w')
            f.writelines(trainlable)

            if assign_testdata == False:
                num = int(len(validation_or_test)/2)
                validationlable = validation_or_test[0:num]
                testlable = validation_or_test[num:]
                test_lable_save_path = os.path.join(lable_save_path, '../test')
                if os.path.exists(test_lable_save_path) == False:
                    os.makedirs(test_lable_save_path)
                f = open(os.path.join(test_lable_save_path, j), 'w')
                f.writelines(testlable)
                label_path['test'].append(test_lable_save_path)
            else:
                validationlable = validation_or_test[0:]

            f = open(os.path.join(validation_lable_save_path, j), 'w')
            f.writelines(validationlable)
            label_path['train'].append(train_lable_save_path)
            label_path['validation'].append(validation_lable_save_path)
    return label_path

def make_lable(label_path,save_path,assign_testdata:bool=False):
    file_name = os.listdir(label_path)
    lable = {}
    for i in file_name:
        file = open(os.path.join(label_path,i))
        lable[i] = []
        lines = file.readlines()
        for line in lines:
            id = line
            lable[i].append(id)
    split_lable = {}
    for i in file_name:
        split_lable[i] = []
        childern_list_len = int(len(lable[i]))+1
        for slice_index in range(0, len(lable[i]), childern_list_len):
            split_lable[i].extend(lable[i][slice_index:slice_index + childern_list_len])

    train_lable_save_path = os.path.join(save_path, 'train')
    validation_lable_save_path = os.path.join(save_path, 'validation')
    test_lable_save_path = os.path.join(save_path, '../test')
    if os.path.exists(train_lable_save_path) == False:
        os.makedirs(train_lable_save_path)
    if os.path.exists(validation_lable_save_path) == False:
        os.makedirs(validation_lable_save_path)
    if os.path.exists(test_lable_save_path) == False:
        os.makedirs(test_lable_save_path)

    for lable_name in file_name:
        if assign_testdata == False:
            train_scale = 0.8
            validation_scale = 0.9
            lable_num = len(split_lable[lable_name])
            train_lable = split_lable[lable_name][:int(lable_num * train_scale)]
            validation_lable = split_lable[lable_name][int(lable_num * train_scale):int(lable_num * validation_scale)]
            test_lable = split_lable[lable_name][int(lable_num * validation_scale):]

            f = open(os.path.join(train_lable_save_path, lable_name), 'w')
            f.writelines(train_lable)
            f.close()
            f = open(os.path.join(validation_lable_save_path, lable_name), 'w')
            f.writelines(validation_lable)
            f.close()
            f = open(os.path.join(test_lable_save_path, lable_name), 'w')
            f.writelines(test_lable)
            f.close()

        elif assign_testdata == True:
            train_scale = 0.8
            lable_num = len(split_lable[lable_name])
            train_lable = split_lable[lable_name][:int(lable_num * train_scale)]
            validation_lable = split_lable[lable_name][int(lable_num * train_scale):]

            f = open(os.path.join(train_lable_save_path, lable_name), 'w')
            f.writelines(train_lable)
            f.close()
            f = open(os.path.join(validation_lable_save_path, lable_name), 'w')
            f.writelines(validation_lable)
            f.close()

    label_path = {}
    label_path['test'] = test_lable_save_path
    label_path['train'] = train_lable_save_path
    label_path['validation'] = validation_lable_save_path
    return label_path

def main():

    parser = argparse.ArgumentParser(description="you should add those parameter")
    parser.add_argument('-data_path', required=True,type=str,help='The data use to train model')
    parser.add_argument('-label_path', required=True,type=str,help='The label use to train model')
    parser.add_argument('-save_path', required=True,type=str,help='The dir use to save train result')
    parser.add_argument('-model_name', default='nanodeep',help='The model to be used')
    parser.add_argument('-device',default='cuda:0',help='The device use to train model')
    parser.add_argument('-model_config',default=None,help='Model hyperparameter settings')
    parser.add_argument('-signal_length',default=5000,type=int,help='How long signals are used for classification, note that the first 1500 signals will be removed')
    parser.add_argument('-epochs',default=30,type=int,help='Number of model iterations')
    parser.add_argument('-batch_size',default=50,type=int,help='The amount of data used to train the model once')
    parser.add_argument('-kfold', default=None, type=int)
    parser.add_argument('-assign_testset_lable', default=None,help='The  data use to test the model')
    parser.add_argument('-assign_testset_data', default=None,help='The  data use to test the model')
    parser.add_argument('--save_best',default=False,action = 'store_true',help='If true will save the best model while train')
    parser.add_argument('--load_to_mem', default=False, action='store_true',help='All train data will be load to memory,it will be faster')
    parser.add_argument('--test_model', default=False, action='store_true')
    opt = parser.parse_args()



    print('-model_name:', opt.model_name)
    print('-data_path:', opt.data_path)
    print('-label_path:', opt.label_path)
    print('-save_path:', opt.save_path)
    print('-device:', opt.device)
    print('-kfold:', opt.kfold)
    print('-model_config:', opt.model_config)
    print('-signal_length:', opt.signal_length)
    print('-epochs:', opt.epochs)
    print('-batch_size:', opt.batch_size)
    print('-assign_testset_lable:', opt.assign_testset_lable)
    print('-assign_testset_data:', opt.assign_testset_data)
    print('-load_data_to_mem:', opt.load_to_mem)
    print('-test_model:', opt.test_model)
    print('-save_best:', opt.save_best)


    if opt.model_config == None:
        model_args = getattr(import_module('read_deep.model_config.defaultconfig'), opt.model_name)
    else:
        ymlfile = open(opt.model_config,'r',encoding='utf-8')
        model_args = yaml.load(ymlfile, Loader=yaml.SafeLoader)


    if opt.assign_testset_data == None and opt.assign_testset_lable == None:
        assign_testdata = False
    else:
        assign_testdata = True


    deepmodel = getattr(import_module('read_deep.model.'+opt.model_name),'model')
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    temp_path = os.path.join(opt.save_path, 'temp')
    lable_save_path = os.path.join(temp_path, 'lable_save_path')
    if os.path.exists(lable_save_path) == False:
        os.makedirs(lable_save_path)
    if opt.kfold !=None:
        label_path = opt.label_path
        label_path = k_fold_lable(label_path, opt.kfold, lable_save_path,assign_testdata=assign_testdata)
        for i in range(opt.kfold):
            print('this ', i, 'fold for ', opt.model_name)
            torch.cuda.set_device(opt.device)

            experiment_save_path = os.path.join(opt.save_path, str(i))
            if not os.path.exists(experiment_save_path):
                os.makedirs(experiment_save_path)
            roc_save_path = os.path.join(experiment_save_path)



            model = deepmodel(**model_args)
            nanopore_gpu = rt_deep(model, opt.signal_length,opt.device)

            nanopore_gpu.load_data(data_path=opt.data_path,
                                   label_path=label_path['train'][i],
                                   dataset='train',
                                   load_to_mem=opt.load_to_mem)
            nanopore_gpu.load_data(data_path=opt.data_path,
                                   label_path=label_path['validation'][i],
                                   dataset='validation',
                                   load_to_mem=opt.load_to_mem)

            nanopore_gpu.train(
                    epochs=opt.epochs,
                    batch_size=opt.batch_size,
                    save_best=opt.save_best,
                    save_path=experiment_save_path
                )

            if opt.test_model == True:
                if assign_testdata == False:
                    nanopore_gpu.load_data(data_path=opt.data_path,
                                           label_path=label_path['test'][i],
                                           dataset='test',
                                           load_to_mem=opt.load_to_mem)
                elif assign_testdata == True:
                    nanopore_gpu.load_data(data_path=opt.assign_testset_data,
                                           label_path=opt.assign_testset_lable,
                                           dataset='test',
                                           load_to_mem=opt.load_to_mem)
                confmat_data,acc_value,lable_all,predict_proba = nanopore_gpu.test_model(opt.batch_size)
                nanopore_gpu.draw_ROC(lable_all,predict_proba,roc_save_path)

    else:
        label_path = make_lable(opt.label_path,lable_save_path)

        torch.cuda.set_device(opt.device)
        experiment_save_path = os.path.join(opt.save_path)
        if not os.path.exists(experiment_save_path):
            os.makedirs(experiment_save_path)
        roc_save_path = os.path.join(experiment_save_path)
        csv_save_path = os.path.join(experiment_save_path, 'acc.csv')

        model = deepmodel(**model_args)
        nanopore_gpu = rt_deep(model, opt.signal_length,opt.device)

        print('loading train data')
        nanopore_gpu.load_data(data_path=opt.data_path,
                               label_path=label_path['train'],
                               dataset='train',
                               load_to_mem=opt.load_to_mem)
        print('loading validation data')
        nanopore_gpu.load_data(data_path=opt.data_path,
                               label_path=label_path['validation'],
                               dataset='validation',
                               load_to_mem=opt.load_to_mem)
        print('load data complete')
        print('start training')
        nanopore_gpu.train(
            epochs=opt.epochs,
            batch_size=opt.batch_size,
            save_best=opt.save_best,
            save_path=experiment_save_path,
        )


        if opt.test_model == True:
            if assign_testdata == False:
                nanopore_gpu.load_data(data_path=opt.data_path,
                                       label_path=label_path['test'],
                                       dataset='test',
                                       load_to_mem=opt.load_to_mem)
            elif assign_testdata == True:
                nanopore_gpu.load_data(data_path=opt.assign_testset_data,
                                       label_path=opt.assign_testset_lable,
                                       dataset='test',
                                       load_to_mem=opt.load_to_mem)
            confmat_data, acc_value, lable_all, predict_proba = nanopore_gpu.test_model(opt.batch_size)
            nanopore_gpu.draw_ROC(lable_all, predict_proba, roc_save_path)
            f = open(csv_save_path,'w')
            writer = csv.writer(f)
            writer.writerows(confmat_data)
            writer.writerow(['accuracy:'+str(acc_value)])
            f.close()

if __name__ == '__main__':
    main()



