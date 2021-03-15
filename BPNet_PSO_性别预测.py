# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 11:08:55 2021

@author: 92102
"""
import pandas as pd
import math
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats
import random

class Initializer():
    # xavier 初始化方法
    def xavier(self, num_neuron_inputs, num_neuron_outputs):
        temp1 = np.sqrt(6) / np.sqrt(num_neuron_inputs + num_neuron_outputs + 1)
        weights = stats.uniform.rvs(-temp1, 2 * temp1, (num_neuron_inputs, num_neuron_outputs))
        return weights


class MetricCalculator:
    def __init__(self, label, predict):
        self.label = label
        self.predict = predict
        assert len(label)==len(predict), "length of label and predict must be equal"
        self.mse = None
        self.rmse = None
        self.mae = None
        self.auc = None

    def get_mse(self):
        self.mse = np.mean(np.sum(np.square(self.label - self.predict),1))

    def get_rmse(self):
        self.rmse = np.sqrt(np.mean(np.sum(np.square(self.label - self.predict), 1)))

    def get_mae(self):
        self.mae = np.mean(np.sum(np.abs(self.label - self.predict),1))

    def get_auc(self):
        prob = self.predict.reshape(-1).tolist()
        label = self.label.reshape(-1).tolist()
        f = list(zip(prob, label))
        rank = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]
        rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]
        posNum = 0
        negNum = 0
        for i in range(len(label)):
            if (label[i] == 1):
                posNum += 1
            else:
                negNum += 1
        self.auc = (sum(rankList) - (posNum * (posNum + 1)) / 2) / (posNum * negNum)

    def print_metrics(self):
        if(self.mse): print("mse: ",self.mse)
        if(self.rmse): print("rmse: ",self.rmse)
        if(self.mae): print("mae: ",self.mae)
        if(self.auc): print("auc: ",self.auc)





class LossFunction():
    # SoftmaxWithLoss函数及其导数的定义
    def softmax_logloss(self, inputs, label):
        temp1 = np.exp(inputs)
        probability = temp1 / (np.tile(np.sum(temp1, 1), (inputs.shape[1], 1))).T
        temp3 = np.argmax(label, 1)  # 纵坐标
        temp4 = [probability[i, j] for (i, j) in zip(np.arange(label.shape[0]), temp3)]
        loss = -1 * np.mean(np.log(temp4))
        return loss

    def der_softmax_logloss(self, inputs, label):
        temp1 = np.exp(inputs)
        temp2 = np.sum(temp1, 1)  # 它得到的是一维的向量;
        probability = temp1 / (np.tile(temp2, (inputs.shape[1], 1))).T
        gradient = probability - label
        return gradient

    def sigmoid_logloss(self, inputs, label):
        probability = np.array([(1.0 / (1 + np.exp(-i))) for i in inputs])
        loss = - np.sum(np.dot(label.T,np.log(probability)+ np.dot((1-label).T,np.log(1-probability)))) / ( len(label))
        return loss

    def der_sigmoid_logloss(self, inputs, label):
        probability = np.array([(1.0 / (1 + np.exp(-i))) for i in inputs])
        gradient = label - probability
        return gradient

    def least_square_loss(self, predict, label):
        tmp1 = np.sum(np.square(label - predict), 1)
        loss = np.mean(tmp1)
        return loss

    def der_least_square_loss(self, predict, label):
        gradient = predict - label
        return gradient


class Losslayer():
    #损失层
    def __init__(self, loss_function_name):
        self.lossfunc = LossFunction()
        self.inputs = 0
        self.loss = 0
        self.grad_inputs = 0
        if loss_function_name == 'SoftmaxLogloss':
            self.loss_function = self.lossfunc.softmax_logloss
            self.der_loss_function = self.lossfunc.der_softmax_logloss
        elif loss_function_name == 'LeastSquareLoss':
            self.loss_function = self.lossfunc.least_square_loss
            self.der_loss_function = self.lossfunc.der_least_square_loss
        else:
            print("wrong loss function")
    def get_label_for_loss(self, label):
        self.label = label

    def get_inputs_for_loss(self, inputs):
        self.inputs = inputs

    def compute_loss(self):
        self.loss = self.loss_function(self.inputs, self.label)

    def compute_gradient(self):
        self.grad_inputs = self.der_loss_function(self.inputs, self.label)

class ActivationFunction():
    def __init__(self):
        pass
    # sigmoid函数及其导数的定义
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def der_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    # tanh函数及其导数的定义
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def der_tanh(self, x):
        return 1 - self.tanh(x) * self.tanh(x)

    # ReLU函数及其导数的定义
    def relu(self, x):
        temp = np.zeros_like(x)
        if_bigger_zero = (x > temp)
        return x * if_bigger_zero

    def der_relu(self, x):
        temp = np.zeros_like(x)
        if_bigger_equal_zero = (x >= temp)  # 在零处的导数设为1
        return if_bigger_equal_zero * np.ones_like(x)

    # Identity函数及其导数的定义
    def identity(self, x):
        return x

    def der_identity(self, x):
        return x

class ActivationLayer():
    #激活层
    def __init__(self, activation_function_name):
        self.actfunc = ActivationFunction()
        if activation_function_name == 'sigmoid':
            self.activation_function = self.actfunc.sigmoid
            self.der_activation_function = self.actfunc.der_sigmoid
        elif activation_function_name == 'tanh':
            self.activation_function = self.actfunc.tanh
            self.der_activation_function = self.actfunc.der_tanh
        elif activation_function_name == 'relu':
            self.activation_function = self.actfunc.relu
            self.der_activation_function = self.actfunc.der_relu
        elif activation_function_name == 'linear':
            self.activation_function = self.actfunc.identity
            self.der_activation_function = self.actfunc.der_identity
        else:
            print('wrong activation function')
        self.inputs = 0
        self.outputs = 0
        self.grad_inputs = 0
        self.grad_outputs = 0

    def get_inputs_for_forward(self, inputs):
        self.inputs = inputs

    def forward(self):
        self.outputs = self.activation_function(self.inputs)

    def get_inputs_for_backward(self, grad_outputs):
        self.grad_outputs = grad_outputs

    def backward(self):
        self.grad_inputs = self.grad_outputs * self.der_activation_function(self.inputs)

class FullyConnectedlayer():
    #全连接层
    def __init__(self, num_neuron_inputs, num_neuron_outputs, batch_size=16,weights_decay=0.001):
        self.num_neuron_inputs = num_neuron_inputs
        self.num_neuron_outputs = num_neuron_outputs
        self.inputs = np.zeros((batch_size, num_neuron_inputs))
        self.outputs = np.zeros((batch_size, num_neuron_outputs))
        self.weights = np.zeros((num_neuron_inputs, num_neuron_outputs))
        self.bias = np.zeros(num_neuron_outputs)
        self.weights_previous_direction = np.zeros((num_neuron_inputs, num_neuron_outputs))
        self.bias_previous_direction = np.zeros(num_neuron_outputs)
        self.grad_weights = np.zeros((batch_size, num_neuron_inputs, num_neuron_outputs))
        self.grad_bias = np.zeros((batch_size, num_neuron_outputs))
        self.grad_inputs = np.zeros((batch_size, num_neuron_inputs))
        self.grad_outputs = np.zeros((batch_size, num_neuron_outputs))
        self.batch_size = batch_size
        self.weights_decay = weights_decay

    def initialize_weights(self, initializer):
        self.weights = initializer(self.num_neuron_inputs, self.num_neuron_outputs)

    # 在正向传播过程中,用于获取输入;
    def get_inputs_for_forward(self, inputs):
        self.inputs = inputs

    def forward(self):
        self.outputs = self.inputs.dot(self.weights)+ np.tile(self.bias, (self.batch_size, 1))

    # 在反向传播过程中,用于获取输入;
    def get_inputs_for_backward(self, grad_outputs):
        self.grad_outputs = grad_outputs

    def backward(self):
        # 求权值的梯度,求得的结果是一个三维的数组,因为有多个样本;
        for i in np.arange(self.batch_size):
            self.grad_weights[i, :] = np.tile(self.inputs[i, :], (1, 1)).T.dot(np.tile(self.grad_outputs[i, :], \
                             (1, 1))) + self.weights * self.weights_decay
        # 求偏置的梯度;
        self.grad_bias = self.grad_outputs
        # 求输入的梯度;
        self.grad_inputs = self.grad_outputs.dot(self.weights.T)

    def update(self, optimizer):
        # 权值与偏置的更新;
        grad_weights_average = np.mean(self.grad_weights, 0)
        grad_bias_average = np.mean(self.grad_bias, 0)
        (self.weights, self.weights_previous_direction) = optimizer(self.weights, grad_weights_average,\
                                            self.weights_previous_direction)
        (self.bias, self.bias_previous_direction) = optimizer(self.bias,grad_bias_average, self.bias_previous_direction)

    def update_batch_size(self,batch_size):
        self.batch_size = batch_size

class Optimizer():
    #优化算法
    def __init__(self, lr, momentum, iteration, gamma, power):
        self.lr = lr
        self.momentum = momentum
        self.iteration = iteration
        self.gamma = gamma
        self.power = power
    # 固定方法
    def fixed(self):
        return self.lr

    # inv方法
    def anneling(self):
        if self.iteration == -1:
            assert False, '需要在训练过程中,改变update_method 模块里的 iteration 的值'
        self.lr = self.lr * np.power((1 + self.gamma * self.iteration), -self.power)
        return self.lr

    # 基于批量的随机梯度下降法
    def batch_gradient_descent_fixed(self, weights, grad_weights, previous_direction):
        direction = self.momentum * previous_direction + self.lr * grad_weights
        weights_now = weights - direction
        return (weights_now, direction)

    def batch_gradient_descent_anneling(self, weights, grad_weights, previous_direction):
        self.lr = self.anneling()
        direction = self.momentum * previous_direction + self.lr * grad_weights
        weights_now = weights - direction
        return (weights_now, direction)

    def update_iteration(self, iteration):
        self.iteration = iteration


class BPNet():
    def __init__(self, optimizer,initializer,batch_size,weights_decay):          
        self.optimizer = optimizer
        self.initializer = initializer
        self.batch_size = batch_size
        self.weights_decay = weights_decay
        self.fc1 = FullyConnectedlayer(7,16,self.batch_size, self.weights_decay)
        self.ac1 = ActivationLayer('sigmoid')
        self.fc2 = FullyConnectedlayer(16,1,self.batch_size, self.weights_decay)
        self.loss = Losslayer("LeastSquareLoss")

    def forward_train(self,input_data, input_label):
        self.fc1.get_inputs_for_forward(input_data)
        self.fc1.forward()
        self.ac1.get_inputs_for_forward(self.fc1.outputs)
        self.ac1.forward()
        self.fc2.get_inputs_for_forward(self.ac1.outputs)
        self.fc2.forward()

        #print("predict label: \n ", np.concatenate((self.fc2.outputs[:10], input_label[:10]), axis=1))
        self.loss.get_inputs_for_loss(self.fc2.outputs)
        self.loss.get_label_for_loss(input_label)
        self.loss.compute_loss()
        #print("loss: ",self.loss.loss)


    def backward_train(self):
        self.loss.compute_gradient()
        self.fc2.get_inputs_for_backward(self.loss.grad_inputs)
        self.fc2.backward()
        self.ac1.get_inputs_for_backward(self.fc2.grad_inputs)
        self.ac1.backward()
        self.fc1.get_inputs_for_backward(self.ac1.grad_inputs)
        self.fc1.backward()

    def predict(self,input_data):
        self.fc1.get_inputs_for_forward(input_data)
        self.fc1.forward()
        self.ac1.get_inputs_for_forward(self.fc1.outputs)
        self.ac1.forward()

        self.fc2.get_inputs_for_forward(self.ac1.outputs)
        self.fc2.forward()
        return self.fc2.outputs

    def eval(self,input_data, input_label):
        self.fc1.update_batch_size(input_data.shape[0])
        self.fc1.get_inputs_for_forward(input_data)
        self.fc1.forward()
        self.ac1.get_inputs_for_forward(self.fc1.outputs)
        self.ac1.forward()
        self.fc2.update_batch_size(input_data.shape[0])
        self.fc2.get_inputs_for_forward(self.ac1.outputs)
        self.fc2.forward()
        print("predict: \n ",self.fc2.outputs[:10])
        print("label: \n", input_label[:10])
        metric = MetricCalculator(label=input_label, predict=self.fc2.outputs)
        metric.get_mae()
        metric.get_mse()
        metric.get_rmse()
        metric.get_auc()
        metric.print_metrics()

    def update(self):
        self.fc1.update(self.optimizer)
        self.fc2.update(self.optimizer)

    def initial(self):
        self.fc1.initialize_weights(self.initializer)
        self.fc2.initialize_weights(self.initializer)
        print('fc1:',self.fc1.initialize_weights(self.initializer))
        print('fc2:',self.fc2.initialize_weights(self.initializer))

class DataHander():
    #数据预处理
    def __init__(self,batch_size):
        self.data_sample = 0
        self.data_label = 0
        self.output_sample = 0
        self.output_label = 0
        self.point = 0  # 用于记住下一次pull数据的地方;
        self.batch_size = batch_size

    def get_data(self, sample, label):  # sample 每一行表示一个样本数据, label的每一行表示一个样本的标签.
        self.data_sample = sample
        self.data_label = label

    def shuffle(self):  # 用于打乱顺序;
        random_sequence = random.sample(range(self.data_sample.shape[0]), self.data_sample.shape[0])
        self.data_sample = self.data_sample[random_sequence]
        self.data_label = self.data_label[random_sequence]

    def pull_data(self):  # 把数据推向输出
        start = self.point
        end = start + self.batch_size
        output_index = np.arange(start, end)
        if end > self.data_sample.shape[0]:
            end = end - self.data_sample.shape[0]
            output_index = np.append(np.arange(start, self.data_sample.shape[0]), np.arange(0, end))
        self.output_sample = self.data_sample[output_index]
        self.output_label = self.data_label[output_index]
        self.point = end % self.data_sample.shape[0]

if __name__ == "__main__": 
    #程序入口
    
    #读取数据
    gender_prediction_data = pd.read_excel("性别预测原始数据.xlsx")
    print("data_shape:", gender_prediction_data.shape)

    data_sample = gender_prediction_data.iloc[:, :-1].values
    data_label = gender_prediction_data.iloc[:, -1].values.reshape(-1,1)
    
    #归一化
    mean = data_sample.mean(axis=0)
    std = data_sample.std(axis=0)
    data_sample = (data_sample-mean)/std
    
    #分割训练集测试集
    data_length = data_label.shape[0]
    train_data_length = int(data_length * 0.8)
    print("train_label_length:",train_data_length)
    data_sample_train, data_sample_test = data_sample[:train_data_length], data_sample[train_data_length:]
    data_label_train, data_label_test = data_label[:train_data_length], data_label[train_data_length:]
    
    #初始化参数设置
    num_iterations = 1000#迭代次数
    lr = 0.001#学习率
    weight_decay = 0.01#权值衰减
    train_batch_size = 16#批次
    test_batch_size = 100
    data_handler = DataHander(train_batch_size)
    opt = Optimizer(lr = lr,momentum = 0.9,iteration = 0,gamma = 0.0005,power = 0.75)
    
    #初始化神经网络权重
    initializer = Initializer()
    
    data_handler.get_data(sample=data_sample_train,label=data_label_train)
    data_handler.shuffle()
    
    bpn = BPNet(optimizer = opt.batch_gradient_descent_anneling, initializer = initializer.xavier, batch_size = 16,\
                weights_decay = 0.001)
    bpn.initial()
    train_error = []
    max_loss = math.inf
    early_stopping_iter = 15
    early_stopping_mark = 0
    for i in range(num_iterations):
        #print('第', i, '次迭代')
        opt.update_iteration(i)
        data_handler.pull_data()
        bpn.forward_train(data_handler.output_sample,data_handler.output_label)
        bpn.backward_train()
        bpn.update()
        train_error.append(bpn.loss.loss)
        if max_loss >  bpn.loss.loss:
            early_stopping_mark = 0
            max_loss = bpn.loss.loss
        if early_stopping_mark > early_stopping_iter:
            break
        early_stopping_mark += 1
    plt.plot(train_error)
    plt.show()
    
    #测试
    bpn.eval(data_sample_test,data_label_test)