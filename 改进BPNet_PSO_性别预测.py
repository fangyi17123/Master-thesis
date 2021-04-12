# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 11:08:55 2021

@author: 92102
"""
import pandas as pd
import math
from matplotlib import pyplot as plt
import numpy as np
import random
import time


class Particle():
    # 初始化
    def __init__(self, x_max, v_max, dim):
        self.Pposition = [random.uniform(-x_max, x_max) for i in range(dim)]  # 粒子的位置
        self.Pv = [random.uniform(-v_max, v_max) for i in range(dim)]  # 粒子的速度
        self.Pbest_position = [0.0 for i in range(dim)]  # 粒子最好的位置
        self.PfitnessValue = []  # 适应度函数值

    def set_Ppos(self, i, value):
        self.Pposition[i] = value

    def get_Ppos(self):
        return self.Pposition

    def set_PbestPosition(self, i, value):
        self.Pbest_position[i] = value

    def get_PbestPosition(self):
        return self.Pbest_position

    def set_Pv(self, i, value):
        self.Pv[i] = value

    def get_Pv(self):
        return self.Pv

    def set_Pfitness_value(self, value):
        self.PfitnessValue = value

    def get_Pfitness_value(self):
        return self.PfitnessValue


class PSO():
    def __init__(self, dim, size, iter_num, x_max, v_max):
        self.C1 = 2
        self.C2 = 2
        self.W = 1#惯性权重
        self.dim = dim  # 粒子的维度
        self.size = size  # 粒子个数
        self.iter_num = iter_num  # 迭代次数
        self.x_max = x_max#当前位置
        self.v_max = v_max  # 粒子最大速度
        self.Gbest_fitness_value = float('Inf')
        self.Gbest_position = [0.0 for i in range(dim)]  # 种群最优位置
        self.Particle_list = [Particle(self.x_max, self.v_max, self.dim) for i in range(self.size)]# 对种群进行初始化

    def set_GbestFitnessValue(self, value):
        self.Gbest_fitness_value = value

    def get_GbestFitnessValue(self):
        return self.Gbest_fitness_value

    def set_GbestPosition(self, i, value):
        self.Gbest_position[i] = value

    def get_GbestPosition(self):
        return self.Gbest_position

    # 更新速度
    def update_vel(self, part):
        for i in range(self.dim):
            vel_value = self.W * part.get_Pv()[i] + self.C1 * random.random() * (part.get_PbestPosition()[i] - \
                    part.get_Ppos()[i]) + self.C2 * random.random() * (self.get_GbestPosition()[i] - part.get_Ppos()[i])
            if vel_value > self.v_max:
                vel_value = self.v_max
            elif vel_value < -self.v_max:
                vel_value = -self.v_max
            part.set_Pv(i, vel_value)

    # 更新位置
    def update_pos(self, part):
        for i in range(self.dim):
            pos_value = part.get_Ppos()[i] + part.get_Pv()[i]
            part.set_Ppos(i, pos_value)        

class MetricCalculator():
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



class Losslayer:
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
        elif loss_function_name == 'sigmoid_logloss':
            self.loss_function = self.lossfunc.sigmoid_logloss
            self.der_loss_function = self.lossfunc.der_sigmoid_logloss
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
        
class ActivationFunction:
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
        return 1   

class ActivationLayer:
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
    def __init__(self, num_neuron_inputs, num_neuron_outputs, batch_size,weights_decay):
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
        # BP求解权值与偏置的更新;
        grad_weights_average = np.mean(self.grad_weights, 0)
        grad_bias_average = np.mean(self.grad_bias, 0)
        (self.weights, self.weights_previous_direction) = optimizer(self.weights, grad_weights_average,\
                                            self.weights_previous_direction)
        (self.bias, self.bias_previous_direction) = optimizer(self.bias,grad_bias_average, \
                                                     self.bias_previous_direction)
        
    def update_PSO(self,uP_positon):
        #PSO求解权值与偏置的更新;
        uP_positon2=list(uP_positon)
        
        key=0  
        for upi in range(0,self.num_neuron_inputs):
            for upj in range(0,self.num_neuron_outputs):
                self.weights[upi][upj]=uP_positon2[key]
                key=key+1
                
        for upi2 in range(0,self.num_neuron_outputs):
            self.bias[upi2]=uP_positon2[key]
            key=key+1
  
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
        self.lr = self.lr * np.power((1 + self.gamma * self.iteration), -self.power)#指数衰减退火
        return self.lr

    def batch_gradient_descent_anneling(self, weights, grad_weights, previous_direction):
        self.lr = self.anneling()
        direction = self.momentum * previous_direction + self.lr * grad_weights
        weights_now = weights - direction
        return (weights_now, direction)

    def update_iteration(self, iteration):
        self.iteration = iteration


class BPNet():
    def __init__(self, batch_size,weights_decay,optimizer,nSampDim,nHidden,nOut):          
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.nSampDim=nSampDim
        self.nHidden = nHidden
        self.nOut = nOut
        self.weights_decay = weights_decay
        self.fc1 = FullyConnectedlayer(self.nSampDim,self.nHidden,self.batch_size, self.weights_decay)
        self.ac1 = ActivationLayer('sigmoid')
        self.fc2 = FullyConnectedlayer(self.nHidden,self.nOut,self.batch_size, self.weights_decay)
        self.ac2 = ActivationLayer('sigmoid')
        self.loss = Losslayer("LeastSquareLoss")
        

    def forward_train(self,input_data, input_label):
        self.fc1.get_inputs_for_forward(input_data)
        self.fc1.forward()
        self.ac1.get_inputs_for_forward(self.fc1.outputs)
        self.ac1.forward()
        self.fc2.get_inputs_for_forward(self.ac1.outputs)
        self.fc2.forward()
        self.ac2.get_inputs_for_forward(self.fc2.outputs)
        self.ac2.forward()

        #print("predict label: \n ", np.concatenate((self.ac2.outputs[:10], input_label[:10]), axis=1))
        self.loss.get_inputs_for_loss(self.ac2.outputs)
        self.loss.get_label_for_loss(input_label)
        self.loss.compute_loss()
        #print("loss: ",self.loss.loss)


    def backward_train(self):
        self.loss.compute_gradient()
        self.ac2.get_inputs_for_backward(self.loss.grad_inputs)
        self.ac2.backward()
        self.fc2.get_inputs_for_backward(self.ac2.grad_inputs)
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
        self.ac2.get_inputs_for_forward(self.fc2.outputs)
        self.ac2.forward()
        return self.ac2.outputs

    def eval(self,input_data, input_label):
        self.fc1.update_batch_size(input_data.shape[0])
        self.fc1.get_inputs_for_forward(input_data)
        self.fc1.forward()
        self.ac1.get_inputs_for_forward(self.fc1.outputs)
        self.ac1.forward()
        self.fc2.update_batch_size(input_data.shape[0])
        self.fc2.get_inputs_for_forward(self.ac1.outputs)
        self.fc2.forward()
        self.ac2.get_inputs_for_forward(self.fc2.outputs)
        self.ac2.forward()
        print("predict: \n ",self.heaviside(self.ac2.outputs[:self.batch_size]))
        print("label: \n", input_label[:self.batch_size])
        metric = MetricCalculator(label=input_label, predict=self.ac2.outputs)
        metric.get_mae()
        metric.get_mse()
        metric.get_rmse()
        metric.print_metrics()
        
        #metric = MetricCalculator(label=input_label, predict=self.heaviside(self.ac2.outputs))
        #metric.get_auc()
        #metric.print_metrics()

    def update(self):
        self.fc1.update(self.optimizer)
        self.fc2.update(self.optimizer)
    
    def update_PSO(self,ff_position):
        uP_positon=list(ff_position)
        self.fc1.update_PSO(uP_positon[0:(self.nSampDim+1)*self.nHidden])
        self.fc2.update_PSO(uP_positon[len(uP_positon)-(self.nHidden+1):len(uP_positon)])   
    
    def heaviside(self,list):
        for i in range(0,len(list)):
            if list[i][0]>0.5:
                list[i][0]=1
            else:
                list[i][0]=0
        return(list)
        
        

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

def main():
    #程序入口，程序运行计时开始
    time_start=time.time()    
    
    #初始化参数设置
    lr = 0.001#学习率
    weight_decay = 0.01#权值衰减
    train_batch_size = 12#训练批次
    train_error = []#粒子群和BP混合求解 
    max_loss = math.inf
    early_stopping_iter = 100
    early_stopping_mark = 0 
    momentum = 0.9
    iteration = 0
    gamma = 0.0005
    power = 0.75
    nSampDim = 7      # 样本维度，输入层神经元个数
    nHidden = 12      # 隐含层神经元个数
    nOut = 1           # 输出层神经元个数
    size = 10    #粒子个数
    x_max = 10      #位置极值
    v_max = 0.5    #速度极值
    num_iterations = 1000#BP迭代次数
    iter_num = 30   #PSO迭代次数


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


    
    opt = Optimizer(lr,momentum,iteration,gamma,power)
    
    data_handler = DataHander(train_batch_size)    
    data_handler.get_data(data_sample_train,data_label_train) 
    data_handler.shuffle()
    
    #神经网络初始化
    bpn = BPNet(train_batch_size,weight_decay,opt.batch_gradient_descent_anneling,nSampDim,nHidden,nOut )    
    
    # BP损失函数改为PSO适应函数
    def fit_fun(position):  # BP损失函数改为PSO适应函数
        ff_position=list(position)
        bpn.update_PSO(ff_position)
        data_handler.pull_data()
        bpn.forward_train(data_handler.output_sample,data_handler.output_label)
        return bpn.loss.loss
    
    #PSO和BP混合求解   
    #PSO初始化，粒子适应值初始化

    pso = PSO((nSampDim+1)*nHidden+(nHidden+1)*nOut , size, iter_num, x_max, v_max)
    for part in pso.Particle_list:
        part.PfitnessValue = fit_fun(part.get_Ppos())
    
    for i in range(1,iter_num):
        print('PSO第', i, '次迭代')
        for part in pso.Particle_list:
            pso.update_vel(part)  # 更新速度
            pso.update_pos(part)  # 更新位置
            value = fit_fun(part.get_Ppos())
            if value < part.get_Pfitness_value():
                part.set_Pfitness_value(value)
                for i in range(pso.dim):
                    part.set_PbestPosition(i, part.get_Ppos()[i])
            if value < pso.get_GbestFitnessValue():
                pso.set_GbestFitnessValue(value)
                for i in range(pso.dim):
                    pso.set_GbestPosition(i, part.get_Ppos()[i])
        print('损失值是：',pso.get_GbestFitnessValue())
        train_error.append(pso.get_GbestFitnessValue())  # 每次迭代完把当前的最优适应度存到损失列表
    for i in range(1,num_iterations):
        print('BP神经网络第', i, '次迭代')
        opt.update_iteration(i)
        data_handler.pull_data()
        bpn.forward_train(data_handler.output_sample,data_handler.output_label)
        bpn.backward_train()
        bpn.update() 
        print('损失值是：',bpn.loss.loss)
        train_error.append(bpn.loss.loss)
        if max_loss >  bpn.loss.loss:
            early_stopping_mark = 0
            max_loss = bpn.loss.loss
        if early_stopping_mark > early_stopping_iter:
            break
        early_stopping_mark += 1
    
    
    # 绘制误差曲线
    plt.plot(train_error)
    plt.show()
    
    #测试
    bpn.eval(data_sample_test,data_label_test)
    
    #运行计时
    time_end=time.time()
    print('time cost',time_end-time_start,'s')

if __name__ == '__main__':
    main()
    


    
