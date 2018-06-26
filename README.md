# Deep Learning Framework
This repository is an attempt to create a deep learning framework to aid in faster learning process for newbies in the deep learning field.<br><br>
The framework is simply an <b>auto-differentiating program</b> which implements the backpropagation algorithm, this is implemented in the <b>graph</b> class of the <b>autogradients.py</b> python file.
The mathematical operations are recorded using the class file <b>Tensor</b>, all preliminary math operations are implemented in class file <b>op</b>.<br>
Check the <b>tutorial.ipynb</b> file for information regarding how the framework can be used.

<br> <br>
The various mathematical operations available listed below:
* op.add(x,y)
* op.subtract(x,y)
* op.multiply(x,y)
* op.divide(x,y)
* op.dot(x,y)
* op.sum(x,axis=None)
* op.mean(x)
* op.reshape(x)
* op.exp(x)
* op.mse(x)
* op.sigmoid(x)
* op.RelU(x)
* op.tanh(x)
* op.softmax(x,axis=None)
* op.crossentropy(x,y)
* op.softmax_crossentropy(x,axis=None)
