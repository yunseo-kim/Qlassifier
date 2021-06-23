# Qlassifier
2021 Quantum Hackathon Korea: Classical-Quantum Hybrid Convolutional Neural Network

 Recently, machine learning technology significantly improved the performance of the image recognizing algorithms. Convolutional Neural Network (CNN) is one of the well-known models. CNN is a deep learning technology using several con- volution filters to extract certain features of the given data. In this challenge, we will construct a Classical-Quantum Hybrid Convolutional Neural Network (HCNN).
 
 Our model applies quantum convolution filters – quanvolution filters – to get nonlinear feature maps. Since researches in machine learning algorithms showed that such nonlinearity could increase accuracy or decrease training times, we can expect some quantum advantage on featuring out such nonlinear characteristics of the data. However, it is not shown in the previous research that such benefit originates in its quantum feature. Therefore we will first carefully construct the quanvolutional neural network proposed from former research. And then, choosing certain circuits of the quanvolution layer and varying methods applied in the layer, we will figure out if there can be a quantum feature that significantly benefits the accuracy or efficiency of the model.
 
 The first step we will change is the encoding step. We can define more distinct intermediate states other than 0 or 1.
 
 Second, we will choose some quantum circuits used for each filter channel. Some of the important factors would be expressibility, entangling capability, circuit cost, and trainability.
 
 Third, the training algorithm would also train the parameters consisting of the quantum circuits. For example, the rotation angle of the Rx gate will be optimized by the back-propagation algorithm.
 
 Varying the filter size, the number of channels and the number of layers would also affect our model's accuracy or efficiency.
 
 We can also define more intermediate values in the decoding step, just like in the first step. However, in real devices, since the only way to know the state is to measure, we would also define the discrete mappings from the result of shots.
 
 Finally, we will test our model on the real quantum device. We will check if every process is executable on the real quantum device.
 
 The performance of our models will be compared with each other and the classical random nonlinear filtered model.
