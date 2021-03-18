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




class Particle:
    # 初始化
    def __init__(self, x_max, vel_max, dim):
        self.__pos = [random.uniform(-x_max, x_max) for i in range(dim)]  # 粒子的位置
        self.__vel = [random.uniform(-vel_max, vel_max) for i in range(dim)]  # 粒子的速度
        self.__bestPos = [0.0 for i in range(dim)]  # 粒子最好的位置
        self.__fitnessValue = fit_fun(self.__pos)  # 适应度函数值

    def set_pos(self, i, value):
        self.__pos[i] = value

    def get_pos(self):
        return self.__pos

    def set_PbestPosition(self, i, value):
        self.__bestPos[i] = value

    def get_PbestPosition(self):
        return self.__bestPos

    def set_vel(self, i, value):
        self.__vel[i] = value

    def get_vel(self):
        return self.__vel

    def set_PbestFitnessValue(self, value):
        self.__fitnessValue = value

    def get_PbestFitnessValue(self):
        return self.__fitnessValue


class PSO:
    def __init__(self, dim, size, iter_num, x_max, vel_max, best_fitness_value=float('Inf'), C1=2, C2=2, W=1):
        self.C1 = C1
        self.C2 = C2
        self.W = W#惯性权重
        self.dim = dim  # 粒子的维度
        self.size = size  # 粒子个数
        self.iter_num = iter_num  # 迭代次数
        self.x_max = x_max#当前位置
        self.vel_max = vel_max  # 粒子最大速度
        self.best_fitness_value = best_fitness_value
        self.best_position = [0.0 for i in range(dim)]  # 种群最优位置
        self.fitness_val_list = []  # 每次迭代最优适应值

        # 对种群进行初始化
        self.Particle_list = [Particle(self.x_max, self.vel_max, self.dim) for i in range(self.size)]

    def set_GbestFitnessValue(self, value):
        self.best_fitness_value = value

    def get_GbestFitnessValue(self):
        return self.best_fitness_value

    def set_GbestPosition(self, i, value):
        self.best_position[i] = value

    def get_GbestPosition(self):
        return self.best_position

    # 更新速度
    def update_vel(self, part):
        for i in range(self.dim):
            vel_value = self.W * part.get_vel()[i] + self.C1 * random.random() * (part.get_PbestPosition()[i] - \
                    part.get_pos()[i]) + self.C2 * random.random() * (self.get_GbestPosition()[i] - part.get_pos()[i])
            if vel_value > self.vel_max:
                vel_value = self.vel_max
            elif vel_value < -self.vel_max:
                vel_value = -self.vel_max
            part.set_vel(i, vel_value)

    # 更新位置
    def update_pos(self, part):
        for i in range(self.dim):
            pos_value = part.get_pos()[i] + part.get_vel()[i]
            part.set_pos(i, pos_value)
        value = fit_fun(part.get_pos())
        if value < part.get_PbestFitnessValue():
            part.set_PbestFitnessValue(value)
            for i in range(self.dim):
                part.set_PbestPosition(i, part.get_pos()[i])
        if value < self.get_GbestFitnessValue():
            self.set_GbestFitnessValue(value)
            for i in range(self.dim):
                self.set_GbestPosition(i, part.get_pos()[i])


    def update(self,fc1_weights,fc1_bias,fc2_weights,fc2_bias,train_error):
        
        for i in range(self.iter_num):
            for part in self.Particle_list:
                self.update_vel(part)  # 更新速度
                self.update_pos(part)  # 更新位置
            self.fitness_val_list.append(self.get_GbestFitnessValue())  # 每次迭代完把当前的最优适应度存到列表
            train_error.append(self.get_GbestFitnessValue())  # 每次迭代完把当前的最优适应度存到列表
            fc1_weights = np.reshape(np.array((self.get_GbestPosition())[0:nSampDim*nHidden]),(nSampDim,nHidden))
            fc1_bias = np.reshape(np.array( (self.get_GbestPosition())[nSampDim*nHidden:((nSampDim+1)*nHidden)]),(nHidden,))
            fc2_weights = np.reshape(np.array((self.get_GbestPosition())[((nSampDim+1)*nHidden):(((nSampDim+1)*nHidden)+nHidden*nOut)]),(nHidden,nOut))
            fc2_bias=np.reshape(np.array((self.get_GbestPosition())[(nSampDim+1)*nHidden+nHidden*nOut:(nSampDim+1)*nHidden+(nHidden+1)*nOut]),(nOut,))
    
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





class LossFunction:
    # SoftmaxWithLoss函数及其导数的定义
    def least_square_loss(self, predict, label):
        tmp1 = np.sum(np.square(label - predict), 1)
        loss = np.mean(tmp1)
        return loss

    def der_least_square_loss(self, predict, label):
        gradient = predict - label
        return gradient


class Losslayer():
    def __init__(self, loss_function_name):
        self.lossfunc = LossFunction()
        self.inputs = 0
        self.loss = 0
        self.grad_inputs = 0
        if loss_function_name == 'LeastSquareLoss':
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

class ActivationLayer():
    #激活层
    def __init__(self, activation_function_name):
        self.actfunc = ActivationFunction()
        if activation_function_name == 'sigmoid':
            self.activation_function = self.actfunc.sigmoid
            self.der_activation_function = self.actfunc.der_sigmoid
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
        (self.bias, self.bias_previous_direction) = optimizer(self.bias,grad_bias_average, \
                                                     self.bias_previous_direction)

    
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

    # inv方法
    def anneling(self):
        if self.iteration == -1:
            assert False, '需要在训练过程中,改变update_method 模块里的 iteration 的值'
        self.lr = self.lr * np.power((1 + self.gamma * self.iteration), -self.power)
        return self.lr

    def batch_gradient_descent_anneling(self, weights, grad_weights, previous_direction):
        self.lr = self.anneling()
        direction = self.momentum * previous_direction + self.lr * grad_weights
        weights_now = weights - direction
        return (weights_now, direction)

    def update_iteration(self, iteration):
        self.iteration = iteration


class BPNet():
    def __init__(self, batch_size,weights_decay,optimizer,initializer,nSampDim,nHidden,nOut):          
        self.optimizer = optimizer
        self.initializer = initializer
        self.batch_size = batch_size
        self.nSampDim=nSampDim
        self.nHidden = nHidden
        self.nOut = nOut
        self.weights_decay = weights_decay
        self.fc1 = FullyConnectedlayer(self.nSampDim,self.nHidden,self.batch_size, self.weights_decay)
        self.ac1 = ActivationLayer('sigmoid')
        self.fc2 = FullyConnectedlayer(self.nHidden,self.nOut,self.batch_size, self.weights_decay)
        self.loss = Losslayer("LeastSquareLoss")
        

    def forward_train(self,input_data, input_label):
        self.fc1.get_inputs_for_forward(input_data)
        self.fc1.forward()
        self.ac1.get_inputs_for_forward(self.fc1.outputs)
        self.ac1.forward()
        self.fc2.get_inputs_for_forward(self.ac1.outputs)
        self.fc2.forward()

        print("predict label: \n ", np.concatenate((self.fc2.outputs[:10], input_label[:10]), axis=1))
        self.loss.get_inputs_for_loss(self.fc2.outputs)
        self.loss.get_label_for_loss(input_label)
        self.loss.compute_loss()
        print("loss: ",self.loss.loss)


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
    
    #初始化参数设置
    num_iterations = 1000#迭代次数
    lr = 0.001#学习率
    weight_decay = 0.01#权值衰减
    train_batch_size = 12#训练批次
    test_batch_size = 100
    train_error = []
    max_loss = math.inf
    early_stopping_iter = 15
    early_stopping_mark = 0 
    momentum = 0.9
    iteration = 0
    gamma = 0.0005
    power = 0.75
    nSampDim = 7      # 样本维度，输入层神经元个数
    nHidden = 28      # 隐含层神经元个数
    nOut = 1           # 输出层神经元个数
    dim = (nSampDim+1)*nHidden+(nHidden+1)*nOut   #粒子维度
    size = 10    #粒子个数
    iter_num = 20   #PSO迭代次数
    x_max = 10      #位置极值
    vel_max = 0.5    #速度极值

    
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
    
    initializer = Initializer()
    
    bpn = BPNet(train_batch_size,weight_decay,opt.batch_gradient_descent_anneling,initializer.xavier,\
                nSampDim,nHidden,nOut )
    bpn.initial()
    
    
    def fit_fun(X):  # 适应函数
        data_handler.pull_data()
        bpn.forward_train(data_handler.output_sample,data_handler.output_label)
        return bpn.loss.loss
    
    #PSO求解
    pso = PSO(dim, size, iter_num, x_max, vel_max)
    pso.update(bpn.fc1.weights,bpn.fc1.bias,bpn.fc2.weights,bpn.fc2.bias,train_error)
    
    
    #BP求解
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
    
    # 绘制误差曲线
    plt.plot(train_error)
    plt.show()
    
    #测试
    bpn.eval(data_sample_test,data_label_test)
    
    
