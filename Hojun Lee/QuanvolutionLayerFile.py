import torch
from torch.autograd import Variable
import numpy as np
from torch.autograd.function import Function
from PQC import *
import torchvision.transforms as T
class QuanvolutionFunction(Function):
    def forward(ctx,inputs, out_channels, kernel_size, quantum_circuit, shift=np.pi/2, weight=None):
        # weight size: (#channels, #params)
        batch_size, in_channels, len_x, len_y = inputs.size()
        len_x = len_x - kernel_size + 1
        len_y = len_y - kernel_size + 1
        if out_channels % in_channels != 0:
            raise Exception("out_channels must be multiple of in_channels(the second dimension of inputs)\n")
        ctx.in_channels = in_channels
        ctx.out_channels = out_channels
        ctx.kernel_size = kernel_size
        ctx.quantum_circuit = quantum_circuit #PQC
        ctx.shift = shift
        features = []
        for input in inputs:
            feature = []
            for i in range(in_channels):
                print(i)
                xys = []
                for x in range(len_x):
                    ys = []
                    for y in range(len_y):
                        # print(x,y)
                        data = input[0, x:x+kernel_size, y:y+kernel_size]
                        ys.append(quantum_circuit.run(data.flatten(),weight[i]))
                    xys.append(ys)
                feature.append(xys)
            features.append(feature)       
        result = torch.tensor(features)
        result.resize_((batch_size,out_channels,len_x,len_y))
        ctx.save_for_backward(inputs, result)
        return result
    def backward(ctx, grad_output):
        input, exp_z = ctx.saved_tensors

        return 0

class QuanvolutionLayer(torch.nn.Module):
    def __init__(self,out_channels,kernel_size,initial_weight=None,backend=None):
        super(QuanvolutionLayer, self).__init__()
        self.out_channels = out_channels
        self.quantum_circuit = QuantumCircuit(backend=backend)
        self.kernel_size = kernel_size
        if initial_weight is None:
            self.weight = torch.rand(1,len(self.quantum_circuit.params))*2*np.pi
        else:
            self.weight = initial_weight
    def forward(self,inputs):
        return QuanvolutionFunction.apply(inputs,self.out_channels,self.kernel_size,
        self.quantum_circuit,np.pi/2,self.weight)
    def update_weight(self,grad_weight,lr=0.001):
        self.weight -= grad_weight * lr
