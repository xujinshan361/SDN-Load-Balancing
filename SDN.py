# _*_ coding:utf-8 _*_
#coding=utf-8
import os
import struct
import numpy as np
import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import torch as th
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data as Data
import pandas as pd
from torch.autograd import Variable

import csv 



	
def data_set(dataset_path):
    csv_file = open(dataset_path,'r')
    datasets = []
    labels = []
    for num in csv_file:
        field = []
        last_one = num.split('\t\t\n')[-1]
        print(last_one)
        for fie in num.split('\t\t'):
            if fie == last_one:
                break
            else:
                field.append(float(fie))
        labels.append(np.array(int(last_one)))
        datasets.append(field)   
    labels = np.array(labels)
    return datasets,labels

class RNN(nn.Module):
	def __init__(self):
		super(RNN, self).__init__()
		self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
			input_size = 31,
			hidden_size=128,         # rnn hidden unit
			num_layers=2,           # number of rnn layer
			batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
		)
		self.out = nn.Linear(128, 2)
	def forward(self, x):
		x = x.reshape(-1,1,31)
		r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
		out = self.out(r_out[:, -1, :])
		return out
def SDN_RNN():
	print(1)
	#with open("D:\DDOSdata\Week1\Friday\csv\outside.csv",'r')as f:
		#reader = csv.reader(f)
		#datas_sdn = []
		#lists_sdn=[]
		#i =1
		#for i in reader:
			#i[1] = i[1].split('.')[0]
			#lists_sdn.append(i[1])
		#print(lists_sdn)
	#for i in range(79199):
		#count = lists_sdn.count(str(i))
		#print(count)
		#datas_sdn.append(count)
	#io = r'D:\data.xls'

	EPOCH = 20            # train the training data n times, to save time, we just train 1 epoch
	BATCH_SIZE = 16
	TIME_STEP = 1          # rnn time step / image height
	LR = 0.001               # learning rate
	
	#df = pd.read_excel(io, usecols=["switch1", "switch2", "switch3", "switch4"], sheet_name=0)
	#df = df[:len(df)//32*32][:]
	#dat = []
	#for i in range(0, df.shape[0]):
		#sum = np.sum(df.iloc[i].values)
		#dat.append(sum)
	#datas_SDN = []	
	#dat = []
	dat = np.random.poisson(lam=3000,size=100000)
	#for i in range(len(datas_sdn)-50):
		#sum = 0
		#for j in range(50):
		#	sum = sum +datas_sdn[i+j]*j
		#	print(sum)
		#dat.append(sum)		

	print(dat)
	datas = []
	lists = []

	#for j in range(2): 
	for i in range(0,len(dat)-32):
		list_a = dat[i:i+31]
		datas.append(list_a) 
		
		#lists.append(dat[i+15]//3000)
		if dat[i+31]>=3050:
			lists.append(1)
		else:
			lists.append(0)
	#print(datas)
	#print(lists)
	datas = np.array(datas)
	#datas = datas/6000
	lists = np.array(lists)
	print(lists)
	print(datas.shape)

	length_train = int(len(datas)*0.8)
		
	train_data = datas[:length_train]
	train_data = th.Tensor(train_data)
	train_data.reshape(-1,1,31)
	
	train_labels = lists[:length_train]
	train_labels = th.Tensor(train_labels)

	datas = th.Tensor(datas)
	test_x = th.unsqueeze(datas,dim=1).type(th.FloatTensor)[length_train:]
	test_x = th.unsqueeze(test_x,dim=1).type(th.FloatTensor)
	test_y = lists[length_train:]
	
	new_train_data = th.utils.data.TensorDataset(train_data,train_labels)
	train_loader = th.utils.data.DataLoader(dataset=new_train_data, batch_size=BATCH_SIZE, shuffle=True,drop_last=False)


	rnn = RNN()	
	loss_func = nn.CrossEntropyLoss()
	optimizer = th.optim.Adam(rnn.parameters(), lr=1e-2)

	for epoch in range(EPOCH):
		for step, (b_x, b_y) in enumerate(train_loader):        # gives batch data
			b_x = b_x.view(-1, 1, 31)              # reshape x to (batch, time_step, input_size)
			b_y = b_y.long()                              #(32,1,17)
			output = rnn(b_x)                               # rnn output
			loss = loss_func(output, b_y)                   # cross entropy loss
			optimizer.zero_grad()                           # clear gradients for this training step
			loss.backward()                                 # backpropagation, compute gradients
			optimizer.step()                                # apply gradients
		test_output = rnn(test_x)                   # (samples, time_step, input_size)
		pred_y = th.max(test_output, 1)[1].data.numpy()
		accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
		print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.4f' % accuracy)
SDN_RNN()

