import os
import re
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
import matplotlib.pyplot as plt
from ont_fast5_api.multi_fast5 import MultiFast5File
from torch.utils.data import dataset
from torch.utils.data import dataloader
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import csv
from read_deep.preprocess import signal_preprocess
class loaddata_from_disk(dataset.Dataset):
    def __init__(self, data_path, label_path, input_length):
        super(loaddata_from_disk, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.lable_name = []
        self.lable = {}
        self.reads_ids = []
        self.data_statistics = {}
        self.input_length = input_length

        all_file = os.listdir(self.label_path)
        all_file = sorted(all_file, key=str.lower)

        for file_name in all_file:
            file_path = os.path.join(self.label_path, file_name)
            if file_name not in self.lable_name:
                self.lable_name.append(file_name)
                self.data_statistics[file_name] = 0
            file = open(file_path, 'r')
            for line in file:
                self.data_statistics[file_name] += 1
                id = re.split(r"[\n,\t, ]", line)[0]
                self.lable[id] = file_name
                self.reads_ids.append(id)
        for index,label in enumerate(self.lable_name):
            print('Remember Label_{} is:{}  '.format(index+1,self.lable_name[index]),end=' ')
        print('\n')
        all_file = os.listdir(self.data_path)
        self.reader_id_index = {}

        pbar = tqdm(total=len(all_file))
        for file_name in all_file:
            file_path = os.path.join(self.data_path, file_name)
            reader = MultiFast5File(file_path, 'r')
            id_list = reader.get_read_ids()
            for id in id_list:
                self.reader_id_index[id] = MultiFast5File(file_path, 'r')
            pbar.update()
        self.error_data = 0

    def data_status(self):
        all_file = os.listdir(self.data_path)
        data_id = []
        lable_id = self.lable.keys()
        for file_name in all_file:
            file_path = os.path.join(self.data_path, file_name)
            reader = MultiFast5File(file_path, 'r')
            id_list = reader.get_read_ids()
            data_id.extend(id_list)
            print(file_name, len(id_list))

        print('lable len:', len(lable_id))
        print('data len:', len(data_id))

        error_data = []
        for id in lable_id:
            if id not in data_id:
                error_data.append(id)
        print('error data num:', len(error_data))

    def __getitem__(self, index):
        id = self.reads_ids[index]
        label = np.zeros(len(self.lable_name))
        label[self.lable_name.index(self.lable[id])] = 1
        signal = self.reader_id_index[id].get_read(id).get_raw_data()
        signal = signal_preprocess(signal)
        if len(signal) > self.input_length:
            signal = signal[0:self.input_length]
        signal = np.pad(signal, ((0, self.input_length - len(signal))), 'constant', constant_values=0)
        signal = signal[np.newaxis,]
        signal = signal.astype(np.float32)
        return signal, label

    def __len__(self):
        return len(self.lable)

class loaddata_from_memory(dataset.Dataset):
    def __init__(self, data_path, label_path, input_length):
        super(loaddata_from_memory, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.lable_name = []
        self.lable = {}
        self.reads_ids = []
        self.data_statistics = {}
        self.input_length = input_length
        self.last_mean = 0
        self.mean_len = 0
        all_file = os.listdir(self.label_path)
        all_file = sorted(all_file, key=str.lower)
        for file_name in all_file:
            file_path = os.path.join(self.label_path, file_name)
            if file_name not in self.lable_name:
                self.lable_name.append(file_name)
                self.data_statistics[file_name] = 0
            file = open(file_path, 'r')

            for line in file:
                self.data_statistics[file_name] += 1
                id = re.split(r"[\n,\t, ]", line)[0]
                self.lable[id] = file_name
                self.reads_ids.append(id)
        for index, label in enumerate(self.lable_name):
            print('Remember Label_{} is:{}  '.format(index + 1, self.lable_name[index]), end=' ')
        print('\n')
        all_file = os.listdir(self.data_path)

        self.reader_raw_data = {}
        pbar = tqdm(total=len(all_file))
        for file_name in all_file:
            file_path = os.path.join(self.data_path, file_name)
            reader = MultiFast5File(file_path, 'r')
            id_list = reader.get_read_ids()
            id_list = list(set(id_list)&set(self.reads_ids))

            for i, id in enumerate(id_list):
                signal = reader.get_read(id).get_raw_data()
                signal = signal_preprocess(signal)
                if len(signal) > self.input_length:
                    signal = signal[1000:self.input_length + 1000]
                signal = np.pad(signal, ((0, self.input_length - len(signal))), 'constant', constant_values=0)
                signal = signal[np.newaxis,]
                signal = signal.astype(np.float32)

                self.reader_raw_data[id] = signal

            pbar.update()
    def __getitem__(self, index):
        id = self.reads_ids[index]
        label = np.zeros(len(self.lable_name))
        label[self.lable_name.index(self.lable[id])] = 1
        signal = self.reader_raw_data[id]

        return signal, label

    def data_status(self):
        all_file = os.listdir(self.data_path)
        data_id = []
        lable_id = self.lable.keys()
        for file_name in all_file:
            file_path = os.path.join(self.data_path, file_name)
            reader = MultiFast5File(file_path, 'r')
            id_list = reader.get_read_ids()
            data_id.extend(id_list)
            print(file_name, len(id_list))

        print('lable len:', len(lable_id))
        print('data len:', len(data_id))

        error_data = []
        for id in lable_id:
            if id not in data_id:
                error_data.append(id)
        print('error data num:', len(error_data))

    def __len__(self):
        return len(self.reads_ids)

class rt_deep:
    def __init__(self, model: nn.Module, input_length: int, device):
        '''

        :param model: 用来分类的模型
        :param signal_length: 输入信号的长度
        '''
        self.model = model
        self.length = input_length

        self.device = torch.device(device)
        self.model.to(self.device)
        self.validation = False
        self.test = False
        print("use:", self.device)

    def load_the_model_weights(self, model_path):
        '''
        :param model_path: 模型参数保存路径
        :return:
        '''
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def signal_classification(self, signal_data: list):
        self.model.eval()
        input = []
        for i in range(len(signal_data)):
            data = signal_data[i].flatten()
            if data.shape[0] < self.length:
                data = np.pad(data, ((0, self.length - data.shape[0])), 'constant', constant_values=0)
                data = data.reshape((1, -1))
                input.append(data)
            else:
                data = data[0:self.length]
                data = data.reshape((1, -1))
                input.append(data)
        input = np.array(input)
        input = input - np.average(input)
        input = torch.from_numpy(input)
        input = input.to(torch.float)
        input = input.to(self.device)
        output = self.model(input)
        output = torch.softmax(output, dim=1)
        output = output.cpu().detach().numpy()
        return output

    def load_data(self,
                  data_path,
                  label_path,
                  dataset: str,  # 'train','validation','test'
                  load_to_mem=False,
                  ):
        if dataset == 'train':
            if load_to_mem:
                self.train_dataset = loaddata_from_memory(data_path, label_path, self.length)
            else:
                self.train_dataset = loaddata_from_disk(data_path, label_path, self.length)

        if dataset == 'validation':
            self.validation = True
            if load_to_mem:
                self.validation_dataset = loaddata_from_memory(data_path, label_path, self.length)
            else:
                self.validation_dataset = loaddata_from_disk(data_path, label_path, self.length)

        if dataset == 'test':
            self.test = True
            if load_to_mem:
                self.test_dataset = loaddata_from_memory(data_path, label_path, self.length)
            else:
                self.test_dataset = loaddata_from_disk(data_path, label_path, self.length)
                self.test_dataloader = dataloader.DataLoader(
                    dataset=self.test_dataset,
                    batch_size=1,
                    shuffle=True
                )

    def train(self,
              epochs: int = 200,
              batch_size=20,
              optim=None,
              loss_fun=None,
              to_validation=True,
              save_best=False,
              save_path=None,
              **kwargs
              ):
        self.train_dataloader = dataloader.DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        self.acc = 0
        if self.validation == True:
            self.validation_dataloader = dataloader.DataLoader(
                dataset=self.validation_dataset,
                batch_size=batch_size,
                shuffle=True
            )

        if loss_fun == None:
            loss_fun = nn.MultiLabelSoftMarginLoss()
        if optim == None:
            optim = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        Evaluation = torchmetrics.Accuracy()

        model_save_path = os.path.join(save_path,'model.pth')
        loss_result_save_path = os.path.join(save_path, 'loss.png')
        metrics_result_save_path = os.path.join(save_path, 'metrics.png')
        val_loss_result_save_path = os.path.join(save_path, 'val_loss.csv')
        train_loss_result_save_path = os.path.join(save_path, 'train_loss.csv')
        val_metrics_result_save_path = os.path.join(save_path, 'val_metrics.csv')
        train_metrics_result_save_path = os.path.join(save_path, 'train_metrics.csv')


        train_history_loss = []
        validation_history_loss = []
        train_history_metrics = []
        validation_history_metrics = []
        print('mean')
        for epoch in range(epochs):
            self.model.train()

            pbar = tqdm(total=self.train_dataloader.__len__())
            pbar.set_description('Train epoch {}:'.format(epoch+1))
            epoch_loss = []
            for step, (input, lable) in enumerate(self.train_dataloader):
                input = input.to(self.device)
                lable = lable.to(self.device)
                logit = self.model(input, **kwargs)
                loss = loss_fun(logit, lable)
                optim.zero_grad()
                loss.backward()
                optim.step()
                lable = lable.to(torch.int)
                lable = lable.to('cpu')
                logit = logit.to('cpu')
                Evaluation(logit, lable)
                pbar.update(1)
                pbar.set_postfix({'Train acc:' : str(np.around(Evaluation.compute().numpy(),4)),'Train loss:' : str(np.around(loss.cpu().detach().numpy(),4))})
                epoch_loss.append(loss.cpu().detach().numpy())
            train_history_loss.append(np.average(epoch_loss))
            train_history_metrics.append(np.around(Evaluation.compute().numpy(),4))

            Evaluation.reset()
            if to_validation == True:
                pbar = tqdm(total=self.validation_dataloader.__len__())
                pbar.set_description('Validation epoch {}:'.format(epoch + 1))
                epoch_loss = []
                for step, (input, lable) in enumerate(self.validation_dataloader):
                    input = input.to(self.device)
                    lable = lable.to(self.device)
                    logit = self.model(input, **kwargs)
                    loss = loss_fun(logit, lable)
                    lable = lable.to(torch.int)
                    lable = lable.to('cpu')
                    logit = logit.to('cpu')
                    Evaluation(logit, lable)
                    pbar.update(1)
                    pbar.set_postfix({'Val acc:': str(np.around(Evaluation.compute().numpy(),4)),'Val loss:': str(np.around(loss.cpu().detach().numpy(),4))})
                    epoch_loss.append(loss.cpu().detach().numpy())
                validation_history_loss.append(np.average(epoch_loss))
                validation_history_metrics.append(np.around(Evaluation.compute().numpy(), 4))
                if save_best == True:
                    now_acc = np.around(Evaluation.compute().numpy(),4)
                    if now_acc > self.acc:
                        self.acc = now_acc
                        print('save the model for acc', self.acc)
                        self.save_model(model_save_path)
                    else:
                        print('the best acc is', self.acc, 'now is ', now_acc)
                else:
                    self.acc = np.around(Evaluation.compute().numpy(),4)
            Evaluation.reset()
        if save_best == False:
            self.save_model(model_save_path)

        plt.figure(figsize=[7, 7], dpi=500)
        plt.plot(train_history_loss,label='Train loss', color='b', lw=3)
        plt.plot(validation_history_loss,label='Validation loss', color='r', lw=3)
        ax = plt.gca()
        bwith = 2
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)
        ax.spines['bottom'].set_linewidth(bwith)
        plt.tick_params(which='major', width=bwith, labelsize=14)
        plt.xlabel('epoches', fontsize=20)
        plt.ylabel('loss', fontsize=20)
        plt.legend(loc="upper right")
        plt.title('Train loss', fontsize=20)
        plt.savefig(loss_result_save_path)

        plt.figure(figsize=[7, 7], dpi=500)
        plt.plot(train_history_metrics, label='Train Accuracy', color='b', lw=3)
        plt.plot(validation_history_metrics, label='Validation Accuracy', color='r', lw=3)
        ax = plt.gca()
        bwith = 2
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)
        ax.spines['bottom'].set_linewidth(bwith)
        plt.tick_params(which='major', width=bwith, labelsize=14)
        plt.xlabel('epoches', fontsize=20)
        plt.ylabel('Accuracy', fontsize=20)
        plt.legend(loc="upper right")
        plt.title('Train metrics', fontsize=20)
        plt.savefig(metrics_result_save_path)

        with open(train_loss_result_save_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['train_loss'])
            writer.writerow(train_history_loss)

        with open(val_loss_result_save_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['validation_loss'])
            writer.writerow(validation_history_loss)

        with open(train_metrics_result_save_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['train_acc'])
            writer.writerow(train_history_metrics)

        with open(val_metrics_result_save_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['validation_acc'])
            writer.writerow(validation_history_metrics)

    def test_model(self,batch_size=50, **kwargs):
        if self.test == False:
            raise ValueError("no test data")
        Evaluation = torchmetrics.Accuracy()
        Confmat = torchmetrics.ConfusionMatrix(num_classes=2)
        confmat_data = []
        self.test_dataloader = dataloader.DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        self.model.eval()
        pbar = tqdm(total=self.test_dataloader.__len__())
        predict_proba = []
        lable_all = []
        for input, lable in self.test_dataloader:
            input = input.to(self.device)
            lable = lable.to(self.device)

            logit = self.model(input, **kwargs)
            lable = lable.to(torch.int)
            lable = lable.detach().to('cpu')
            logit = logit.detach().to('cpu')
            Evaluation(logit, lable)
            confmat_data.append(Confmat(torch.argmax(logit, dim=1), torch.argmax(lable, dim=1)).numpy())
            pbar.update(1)
            pbar.set_postfix({'Val acc:': str(Evaluation.compute().numpy()),})
            lable_all.extend(list(lable.numpy()))
            predict_proba.extend(list(logit.numpy()))
        acc_value = Evaluation.compute()
        confmat_data = list(np.sum(np.array(confmat_data), axis=0))
        lable_all = np.array(lable_all)
        predict_proba = np.array(predict_proba)
        lable_all = lable_all.astype(np.int)
        acc_value = acc_value.numpy()
        return confmat_data,acc_value,lable_all,predict_proba


    def draw_ROC(self, lable_all, predict_proba,save_path, **kwargs):

        roc_img_save_path = os.path.join(save_path, 'roc.pdf')
        roc_csv_save_path = os.path.join(save_path, 'roc.csv')
        fpr, tpr, _ = roc_curve(lable_all[:, 0], predict_proba[:, 0])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=[7, 7], dpi=500)
        plt.plot(fpr, tpr, color='b',
                 lw=3, label='ROC curve  ' + '(area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='r', lw=3, linestyle='--')
        ax = plt.gca()
        bwith = 2
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)
        ax.spines['bottom'].set_linewidth(bwith)
        plt.tick_params(which='major', width=bwith, labelsize=14)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.legend(loc="lower right")
        plt.savefig(roc_img_save_path)
        with open(roc_csv_save_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['fpr'])
            writer.writerow(fpr)
            writer.writerow(['tpr'])
            writer.writerow(tpr)
            writer.writerow(['roc_auc'])
            roc_auc = [roc_auc]
            writer.writerow(roc_auc)

    def data_imformation(self):
        self.train_dataset.data_status()

