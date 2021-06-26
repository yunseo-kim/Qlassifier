import torch
from torch.autograd import Variable
import numpy as np
from torch.autograd.function import Function
from PQC import *
class QuanvolutionFunction(Function):
    def forward(ctx,inputs,in_channels, out_channels, kernel_size, quantum_circuit, shift=np.pi/2, weight=None):
        ctx.in_channels = in_channels
        ctx.out_channels = out_channels
        ctx.kernel_size = kernel_size
        ctx.quantum_circuit = quantum_circuit #PQC
        ctx.shift = shift
        # weight size: (#channels, #params)
        _, _, len_x, len_y = inputs.size()
        len_x = len_x - kernel_size + 1
        len_y = len_y - kernel_size + 1
        features = []
        for input in inputs:
            feature = []
            for i in weight.size()[0]:
                xys = []
                for x in range(len_x):
                    ys = []
                    for y in range(len_y):
                        data = input[0, x:x+kernel_size, y:y+kernel_size]
                        ys.append(quantum_circuit.run(data,weight[i]))
                    xys.append(ys)
                feature.append(xys)
            features.append(feature)       
        result = torch.tensor(features)

        ctx.save_for_backward(inputs, result)
        return result
    def backward():
        return 0

class QuanvolutionLayer(torch.nn.Module):
    def __init__(self):
        super(QuanvolutionLayer, self).__init__()

    def forward(self,inputs):
        return QuanvolutionFunction.apply()
