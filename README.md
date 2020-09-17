# NuralNet-from-scratch
This is a project in which I create one hidden layered nural network from scratch using numpy and matplotlib.<br/>
<br/>![image](one_layered_network.jpeg)
<br/><br/>As shown in the image there are 'm' nodes in input, 'n' nodes in the hidden layer and an output node.
<br/>The reason to use a neural network is when the data needs a non linear function to fit, so the most primary example we can use is Logic Gates (It is the Hello World for neural nets).
<br/>So for example I am taking XOR Gate as training data. You can feed your own data to this network.
<br/>![image_table](TRUTH-TABLE-1.jpg)
<br/><br/> Now for this example we have 2 inputs so we will set input_nodes = 2, hidden_nodes = 3(can be any number for you), output_nodes = 1(because the output value can be either 0 or 1)
<br/><br/>So what actually happens here??
<br/>First the training data(here the data and y) is fed to the input layer of network. Each node in the layer or the network has a math function, that predicts a hypotheses by given formula.

## Y = W<sup>T</sup>X + b
Here *W* is the weight and *b* is the bias.

Then Y is given to an activation function here we are using "sigmoid function" 
<br/>![image](sigmoid.jfif)
<br/>The output of sigmoid function is represented as 'a', and as this is the first activated value we will represent it by a<sup>[1]</sup>
<br/><br/>Sigmoid Function will give the output ranging between [0, 1].
<br/><br/>By doing this to input_layer, these same steps are being repeated in the hidden layer but here the "X" is the a<sup>[1]</sup>.
<br/>And the output_layer will give some output ranging between [0, 1], so now we have to check whether the answer is right or wrong and if wrong how much we are wrong to predict.
<br/>To find the error in our prediction and actual output is called as *Loss Function*.
<br/>And *Loss Function* is represented by **J**.
<br/>![image](Loss.png)

<br/>Everything that happened till now is called as ***Forward Propagation*** that is because our flow was from input to output. But now we have the loss and we have to improve our prediction so we have to tune our Weights and Biases by going towards Input Layer and tuning all the parameters.
## Back Propagation:
As we have currently worked on the output_layer, we will persuit to the hidden_layer (we have only one hidden layer) and perform *Gradient Descent*.
<br/>![image](Grad_desc.png)
