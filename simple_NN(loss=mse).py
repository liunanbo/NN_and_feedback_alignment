import numpy as np
import pandas as pd



#mini batch
def minibatcher(X, y, batch_size, shuffle=False):
   assert X.shape[0] == y.shape[0]
   n_samples = X.shape[0]
   if shuffle:
      idx = np.random.permutation(n_samples)
   else:
      idx = list(range(n_samples))
   for k in range(int(np.ceil(n_samples/batch_size))):
      from_idx = k*batch_size
      to_idx = (k+1)*batch_size
      yield X[idx[from_idx:to_idx], :], y[idx[from_idx:to_idx], :]

#Sigmoid function and Derivative
def Sigmoid(x,deriv=False):
    if deriv==True:
        return x*(1-x)
    return 1/(1+np.exp(-x))


#softmax function
def softmax(x):
    exps=np.exp(x)
    return exps/exps.sum(axis=1,keepdims=True)


#Negative log loss function
def loss(x,y):
    loss=np.sum(-y*np.log(x)-(1-y)*np.log(1-x))
    return loss

#Mean squared error loss function
def mse(x,y):
    loss=np.mean( np.power(np.subtract(x,y),2))
    return loss


#Feed Forward
def feed_Forward(X,W,B):
    #Append Input layer
    L=[X]
    num_layer=len(W)
    #Append Hidden layer
    for i in range(num_layer-1):
        L.append(Sigmoid(np.dot(L[-1],W[i])+B[i]))
    #Append output layer
    L.append(Sigmoid(np.dot(L[-1],W[-1])+B[-1]))
    return L

#feed backward and Update weights and Bias
def back_propagation(L,Output_error,W,B,learning_rate):
    num_layer=len(W)
    L_delta=[Output_error*Sigmoid(L[-1],deriv=True)]
    for i in reversed(range(1,num_layer)):
        temp_error=L_delta[0].dot(W[i].T)
        temp_delta=temp_error*Sigmoid(L[i],deriv=True)
        L_delta.insert(0,temp_delta)

    #Update weights and Bias
    for i in range(num_layer):
        W[i]-=learning_rate*L[i].T.dot(L_delta[i])
        B[i]-=learning_rate*L_delta[i].sum(axis=0)

# Compute accuracy
def find_acc(W,B,x,y):
       pred=feed_Forward(x,W,B)[-1]
       pred=np.argmax(pred,1)
       gt=np.argmax(y,1)
       acc=np.mean(pred==gt)
       return acc

#training step
def NN_train(X,Y,hidden_layer_node_list=[4,3],epoch=100,learning_rate=1,batch_size=1000):
    #Initialize synapses weight abd Bias
    W=[]
    B=[]
    for i in range(len(hidden_layer_node_list)):
        if i==0:
            W.append(np.random.uniform(-1,1,size=(X.shape[1],hidden_layer_node_list[i])))
        else:
            W.append(np.random.uniform(-1,1,size=(hidden_layer_node_list[i-1],hidden_layer_node_list[i])))
        B.append(np.random.random((1,hidden_layer_node_list[i])))
    W.append(np.random.uniform(-1,1,size=(hidden_layer_node_list[-1],Y.shape[1])))
    B.append(np.random.random((1,Y.shape[1])))

    for j in range(epoch):
        error=[]
        for mb in minibatcher(X, Y, batch_size):
            L=feed_Forward(mb[0],W,B)
            Output_error=L[-1]-mb[1]
            error.append(mse(L[-1],mb[1]))
            #update weights and Bias
            back_propagation(L,Output_error,W,B,learning_rate)
        print('Epoch '+str(j)+' Loss function score: '+str(np.mean(error))+' Accuracy is: '+str(find_acc(W,B,X,Y)))
    return W,B



#load  data Frame
df=pd.read_csv('/Users/liunanbo/Desktop/MSEUMAP_3.csv')
# input data
x=df.iloc[:,16:16+784].values
#output data
gt=df['GT_Label'].values
#one-hot coding
y=[]
for i in gt:
    temp=np.array([0]*10)
    temp[int(i)]=1
    y.append(temp)
y=np.array(y)




W,B=NN_train(x,y,[64,32,20],100,0.01,batch_size=1000)

print(find_acc(W,B,x,y) )









