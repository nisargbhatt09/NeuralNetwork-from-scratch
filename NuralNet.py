import numpy as np
import matplotlib.pyplot as plt

#For example let's consider it as an OR problem.

#Here "data" is the input and "Y" is the correct Output.
data = [[0,0], [0,1], [1,0], [1,1]]
X = np.array(data)
X = X.T
Y = [[0,1,1,1]]
Y = np.array(Y)

# here set the number of neurons in each layer.
def set_neurons():
    print("Enter number of neurons of Input Layer: ")
    # Take x_n as 2 because we are doing "OR Problem".
    x_n = int(input())
    
    print("Enter number of neurons of Hidden Layer: ")
    # h_n can be any number but 2/3 is more optimised.
    h_n = int(input())
    print("Enter number of neurons of Output Layer: ")
    # o_n is 1 as there is only one possible output (0/1).
    o_n = int(input())
    neurons = {"x_n":x_n, "h_n":h_n, "o_n":o_n}
    return(neurons)

neurons = set_neurons()

def set_param():
    x_n = neurons["x_n"]
    h_n = neurons["h_n"]
    o_n = neurons["o_n"]
    # w1, b1, w2, b2 are the parameters of the network.
    # As there is only one hidden layer we are having w1, b1 and w2, b2.
    w1 = np.random.randn(h_n, x_n)
    b1 = np.zeros((h_n,1))
    w2 = np.random.randn(o_n, h_n)
    b2 = np.zeros((o_n,1))
    return w1, b1, w2, b2

w1, b1, w2, b2 = set_param()

# I am using sigmoid function as activation function here.
def sigmoid(z):
    a_sig = 1/(1+np.exp(-z))
    return a_sig

#Forward Propagation
def forward_prop(X_para, w1, w2):
    z1 = np.dot(w1, X_para)+b1
    a1 = sigmoid(z1)
    z2 = np.dot(w2, a1)+b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

z1, a1, z2, a2 = forward_prop(X, w1, w2)

# m is the size of input data.
m = Y.shape[1]

# Find Cost/Loss
def find_cost(Y, z1, a1, z2, a2):
    Loss = (np.multiply(Y, np.log(a2)))+(np.multiply((1-Y), np.log(1-a2)))
    cost = -1*(1/m)*np.sum(Loss)
    cost = float(np.squeeze(cost))
    return(cost)

cost = find_cost(Y, z1, a1, z2, a2)

# Back Propagation
def back_prop(z1, a1, z2, a2, w1, b1, w2, b2):
    dz2 = (1/m)*(a2-Y)
    dw2 = (1/m)*(np.dot(dz2, a1.T))
    db2 = (1/m)*np.sum(dz2, axis = 1, keepdims = True)
    dz1 = np.dot(w2.T, dz2) * a1*(1 - a1)
    dw1 = (1/m)*np.dot(dz1, X.T)
    db1 = (1/m)*np.sum(dz1, axis=1, keepdims = True)
    return dz2, dw2, db2, dz1, dw1, db1

dz2, dw2, db2, dz1, dw1, db1 = back_prop(z1, a1, z2, a2, w1, b1, w2, b2)

# Update The Parameters After The Back Propagation
def update_parameters(dz2, dw2, db2, dz1, dw1, db1, w1, b1, w2, b2, learning_rate = 0.1):   
    w2 = w2-learning_rate*dw2
    b2 = b2-learning_rate*db2
    w1 = w1-learning_rate*dw1
    b1 = b1-learning_rate*db1
    return(w1, b1, w2, b2)

w1, b1, w2, b2 = update_parameters(dz2, dw2, db2, dz1, dw1, db1, w1, b1, w2, b2)
X_temp = X
iterations = 15000
costl = []

# Real Action Takes Part Here
for i in range(iterations):
    z1, a1, z2, a2 = forward_prop(X_temp, w1, w2)
    cost = (-1/m)*np.sum(Y*np.log(a2)+(1-Y)*np.log(1-a2))
    costl.append(cost)
    dz2, dw2, db2, dz1, dw1, db1 = back_prop(z1, a1, z2, a2, w1, b1, w2, b2)
    w1, b1, w2, b2 = update_parameters(dz2, dw2, db2, dz1, dw1, db1, w1, b1, w2, b2)

plt.plot(costl)
plt.show()
print(costl[len(costl)-2]-costl[len(costl)-1])

# Taking Input From User
print("Enter First Input: ")
inp1 = int(input())
print("Enter Second Input: ")
inp2 = int(input())
ls = [inp1, inp2]
print(ls)
inp = np.array(ls)
inp = inp.T
inp = np.array(inp)
inp = np.reshape(inp, (2,1))
print("inp",np.shape(inp))
z1, a1, z2, a2 = forward_prop(inp, w1, w2)
print("A2=", a2)
