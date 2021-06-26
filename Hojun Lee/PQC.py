import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import *

class QuantumCircuit:
    def __init__(self, n_qubits=9, backend=None, shots=1024):
        qr = qiskit.QuantumRegister(n_qubits)
        cr = qiskit.ClassicalRegister(1)
        self.circuit = qiskit.QuantumCircuit(qr,cr)
        self.params = qiskit.circuit.ParameterVector('θ',3*n_qubits)
        self.n_qubits = n_qubits
        self.output_channels = n_qubits

        ## circuit
        for i in range(n_qubits):
            self.circuit.rx(self.params[i],i)
        for i in range(n_qubits):
            self.circuit.rz(self.params[i+n_qubits],i)
        for i in range(n_qubits-1):
            self.circuit.crx(self.params[i+2*n_qubits],i,i+1)
        self.circuit.crx(self.params[3*n_qubits-1],n_qubits-1,0)

        self.backend = backend
        self.shots = shots
    
    def run(self,params):
        output = []
        for j in range(self.output_channels):
            t_qc = transpile(self.circuit,
                         self.backend)
            circ = t_qc.bind_parameters({self.params: params})
            circ.measure(j,0)
            job = qiskit.execute(circ, backend=self.backend,shots=self.shots)
            result = job.result().get_counts()
            print(result)
            counts = np.array(list(result.values()))
            states = np.array(list(result.keys())).astype(float)
            # Compute probabilities for each state
            probabilities = counts / self.shots
            # Get state expectation
            expectation = np.sum(states * probabilities)
            output.append(expectation)
        
        return output