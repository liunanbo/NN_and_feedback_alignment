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


#negative log loss function
def loss(x,y):
    loss=np.sum(-y*np.log(x))
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
    L.append(softmax(np.dot(L[-1],W[-1])+B[-1]))    
    return L
        
#feed backward and Update weights and Bias          
def Direct_feedback_alignment(L,Output_error,W,B,FA_W,learning_rate):
    num_layer=len(W)
    L_delta=[Output_error]
    for i in reversed(range(1,num_layer)):
        #only use last layer output error to update weights
        temp_error=L_delta[-1].dot(FA_W[i].T)
        temp_delta=temp_error*Sigmoid(L[i],deriv=True)
        L_delta.insert(0,temp_delta)
        
    #Update weights and Bias
    for i in range(num_layer):
        W[i]-=learning_rate*L[i].T.dot(L_delta[i])
        B[i]-=learning_rate*L_delta[i].sum(axis=0)
    
        
    
#training step
def DFANN_train(X,Y,hidden_layer_node_list=[4,3],epoch=100,learning_rate=1,batch_size=1000):
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
    FA_W=[np.random.uniform(-0.5,0.5,size=(w.shape[0],W[-1].shape[1])) for w in W]
    for j in range(epoch):
        error=[]
        for mb in minibatcher(X, Y, batch_size): 
            L=feed_Forward(mb[0],W,B)
            Output_error=L[-1]-mb[1]
            error.append(loss(L[-1],mb[1]))
            #update weights and Bias
            Direct_feedback_alignment(L,Output_error,W,B,FA_W,learning_rate)
        print('Epoch '+str(j)+' Loss function score: '+str(np.mean(error)))
    return W,B


 
#load  data Frame  
df=pd.read_csv('/Users/tempadmin/Desktop/MSEUMAP_3.csv')
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




#Train Direct Feedback alignment NN
W,B=DFANN_train(x,y,[64,32,20],300,0.003,batch_size=1000)
pred=feed_Forward(x,W,B)[-1]  
pred=np.argmax(pred,1)   
    
print(np.mean(pred==gt))    
    
    
    
    
    
    
    
    
    
    
    
    
    
