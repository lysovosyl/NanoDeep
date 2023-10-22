#%%
import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_pass_1(nn.Module):
    def __init__(self,in_feature,n_feature_maps=20):
        super(conv_pass_1, self).__init__()
        self.Conv1D = nn.Conv1d(in_channels=in_feature,out_channels=n_feature_maps,kernel_size=1,stride=1,padding=0)
        self.BatchNormalization = nn.BatchNorm1d(n_feature_maps)
        self.Activation = nn.ReLU()
    def forward(self, x):
        y = self.Conv1D(x)
        y = self.BatchNormalization(y)
        y = self.Activation(y)
        return y

class conv_pass_2(nn.Module):
    def __init__(self,in_feature,n_feature_maps=20,strides=1):
        super(conv_pass_2, self).__init__()
        self.Conv1D = nn.Conv1d(in_channels=in_feature,out_channels=n_feature_maps,kernel_size=3,stride=strides,padding=0)
        self.BatchNormalization = nn.BatchNorm1d(n_feature_maps)
        self.Activation = nn.ReLU()
    def forward(self, x):
        y = self.Conv1D(x)
        y = self.BatchNormalization(y)
        y = self.Activation(y)
        return y

class conv_pass_3(nn.Module):
    def __init__(self,in_feature,n_feature_maps=20):
        super(conv_pass_3, self).__init__()
        self.Conv1D = nn.Conv1d(in_channels=in_feature,out_channels=n_feature_maps,kernel_size=1,stride=1,padding=0)
        self.BatchNormalization = nn.BatchNorm1d(n_feature_maps)
        self.Activation = nn.ReLU()
    def forward(self, x):
        y = self.Conv1D(x)
        y = self.BatchNormalization(y)
        y = self.Activation(y)
        return y

class make_layer(nn.Module):
    def __init__(self,in_feature,filters, blocks, strides=1):
        super(make_layer, self).__init__()
        filter_1 = 20
        self.down_sample = None
        if strides != 1 or filter_1 != filters:
            self.down_sample = True

        self.conv_pass_layer_1 = conv_pass_1(in_feature=in_feature,n_feature_maps=filters)
        self.conv_pass_layer_2 = conv_pass_2(in_feature=filters,n_feature_maps=filters,strides=strides)
        self.Conv1D = nn.Conv1d(in_channels=filters,out_channels=filters,kernel_size=1,padding=0,stride=1)
        self.BatchNormalization = nn.BatchNorm1d(num_features=filters)

        self.down_sample_Conv1D = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=1, padding=0, stride=strides)
        self.down_sample_BatchNormalization = nn.BatchNorm1d(num_features=filters)

        self.Activation = nn.ReLU()

        if self.down_sample:
            filter_1 = filters

        self.conv_pass = nn.ModuleList()
        for _ in range(1, blocks):
            self.conv_pass.append(conv_pass_1(filters, filters))
            self.conv_pass.append(conv_pass_2(filters, filters))
            self.conv_pass.append(conv_pass_3(filters, filters))

    def forward(self, x):
        y = self.conv_pass_layer_1(x)
        y = self.conv_pass_layer_2(y)
        y = self.Conv1D(y)
        y = self.BatchNormalization(y)
        if self.down_sample:
            y = self.down_sample_Conv1D(y)
            y = self.down_sample_BatchNormalization(y)
        y = self.Activation(y)
        for layer in self.conv_pass:
            y = layer(y)
        return y

#%%
class model(nn.Module):
    def __init__(self,**kwargs):
        super(model, self).__init__()
        input_shape = kwargs['input_shape']
        self.nb_classes = kwargs['nb_classes']
        self.is_train = kwargs['is_train']

        self.Conv1D = nn.Conv1d(in_channels=1,out_channels=20,kernel_size=19,padding=0,stride=3)
        self.BatchNormalization = nn.BatchNorm1d(num_features=20)
        self.Activation = nn.ReLU()
        self.MaxPooling1D = nn.MaxPool1d(kernel_size=2,padding=0,stride=2)

        self.Dropout_1 = nn.Dropout(0.1)
        self.make_layer_1 = make_layer(in_feature=20,filters=20, blocks=2)
        self.Dropout_2 = nn.Dropout(0.1)
        self.make_layer_2 = make_layer(in_feature=20,filters=30, blocks=2, strides=2)
        self.Dropout_3 = nn.Dropout(0.1)
        self.make_layer_3 = make_layer(in_feature=30,filters=45, blocks=2, strides=2)
        self.Dropout_4 = nn.Dropout(0.1)
        self.make_layer_4 = make_layer(in_feature=45,filters=67, blocks=2, strides=2)

        self.AveragePooling1D = nn.AdaptiveAvgPool1d(1)
        self.Flatten = nn.Flatten()

        self.Dense = nn.Linear(in_features=67,out_features=self.nb_classes)
    def forward(self,x:torch.tensor,**kwargs):
        y = self.Conv1D(x)
        y = self.BatchNormalization(y)
        y = self.Activation(y)
        y = self.MaxPooling1D(y)
        y = self.Dropout_1(y)
        y = self.make_layer_1(y)
        y = self.Dropout_2(y)
        y = self.make_layer_2(y)
        y = self.Dropout_3(y)
        y = self.make_layer_3(y)
        y = self.Dropout_4(y)
        y = self.make_layer_4(y)
        y = self.AveragePooling1D(y)
        y = self.Flatten(y)
        if self.is_train:
            noise = torch.randn(y.shape)
            noise = noise*10
            noise = noise.to(y.device)
            x = torch.cat((y, noise), dim=0)
        else:
            x = y

        y = self.Dense(x)
        return y

# #%%
#
# signal = torch.randn((100,1,5000))
# signal = signal.to('cuda:1')
# kwargs = {"input_shape": 3, "nb_classes": 2, "is_train": False}
# mymodel = model(**kwargs)
# mymodel = mymodel.to('cuda:1')
# y = mymodel(signal)
# print(y.shape)
