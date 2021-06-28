# Qlassifier
## Classical-Quantum Hybrid Convolutional Neural Network
**[2021 Quantum Hackathon Korea](https://qhackathon.kr/) Participant**

### Mentor: [Prof. Daniel K. Park](http://qhackathon.kr/박경덕-교수님/)
### Team Members
| name | github |
|------|--------|
|Boseong Kim|[@BStar14](https://github.com/BStar14)
|Dongok Kim|[@JadeKim](https://github.com/JadeKim)
|Hojun Lee|[@String137](https://github.com/String137)
|Jaehoon Hahm|[@Jaehoon-zx](https://github.com/Jaehoon-zx)
|Yunseo Kim|[@Yunseo47](https://github.com/Yunseo47)

### Introduction
In this challenge, we will construct a Classical-Quantum Hybrid Convolutional Neural Network(HCNN).  
Our model applies quantum convolution filters - quanvolution filters - to get nonlinear feature maps. Since research in machine learning algorithms showed that such nonlinearity could increase accuracy or decrease training times, we can expect some quantum advantage to feature such nonlinear characteristics of the data. However, It is not shown in the previous research that such benefit originates in its *quantum feature*. 
Therefore we will first carefully construct the quanvolutional neural network proposed from former research. And then, choosing certain circuits of the quanvolutional layer and varying methods applied in the layer, we will figure out if there could exist a *quantum feature* in the classical dataset that significantly benefits the accuracy or efficiency of the model.

### What are the benefits of neural networks running on quantum machines?
For now, the error rate of quantum computers is higher than that of classical computers. Therefore, NISQ(Noisy Intermediate-Scale Quantum) algorithms have been proposed to leverage the limited available resources to perform classically challenging tasks.  
Neural networks based on parameterized quantum circuits are relatively robust to gate errors, so it is expected to obtain meaningful results with NISQ computers. The quantum neural network is known to have better expressibility and trainability than the classical neural network, and there are some reported cases in which the quantum neural network achieves higher accuracy even when the number of parameters is reduced exponentially compared to the classical neural network. In particular, if the data to be classified using machine learning is a quantum state, the difference is more pronounced.

### Problem statement
1. **Encoding / decoding methodology**  
We will define intermediate states other than 0 or 1. We will also consider if the decoding process is executable since the only way to know about the final state is to measure it, especially in a real device.
2. **Quantum circuit selection**  
We will choose 4 or 5 candidates from available quantum circuit templates, taking into account the expressibility, entangling capability, circuit cost, and trainability.
3. **Experiment with diifferent number of quanvolution channels**  
Each quanvolution channel extracts a distinct feature of the input data. Adding more quanvolutional filters would increase model performance. Nevertheless, too many channels lead to overfitting issue. We will choose the appropriate number of channels that get the most efficient model.
4. **Training Quanvolution layer**  
The training algorithm would also train the parameters consisting of the quantum circuits. For example, the rotation angle of the Rx gate will be optimized by the back-propagation algorithm. Qiskit gradient framework would enable us to optimize the circuit parameters continually in the quantum device.

### Test and Comparison
We will test our model on the IonQ device. The whole result from our models and the classical random nonlinear filtered model will be compared to each other. Once we found the quantum advantage, we will figure out what kinds of features did the quanvolution layers extracted and how the parameters matched each feature to the result.

### Reference
- Maxwell Henderson, et al. Quanvolutional neural networks: powering image recognition with quantum circuits. Quantum Machine Intelligence, 2(1):2, June 2020.
