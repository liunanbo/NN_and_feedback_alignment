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

def mse(x,y):
    loss=np.mean( np.power(np.subtract(x,y),2))
    return loss

#Adam Descent Optimizer
class AdamOptimizer(object):

    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, name='Adam'):
        self.initial_LRN = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.name = name

        self.Mt_1 = 0
        self.Vt_1 = 0
        self.Mt = None
        self.Vt = None
        self.t = 0    #Step Index
        self.Mt_hat=None
        self.Vt_hat=None

    def update(self, gradients):
        '''

        :param gradients: Feature gradients
        :return: Feature Delta used to update current feature value
        '''
        #increment iteration step t
        self.t += 1
        # Update Mt and Vt
        self.Mt = self.beta1 * self.Mt_1 + (1 - self.beta1) * gradients
        self.Vt = self.beta2 * self.Vt_1 + (1 - self.beta2) * np.power(gradients, 2)

        # Update Mt-1 and Vt-1
        self.Mt_1 = self.Mt
        self.Vt_1 = self.Vt

        self.Mt_hat = np.divide(self.Mt,(1-self.beta1 ** self.t))
        self.Vt_hat = np.divide(self.Vt,(1-self.beta2 ** self.t))


        updated_grad = np.divide(self.Mt_hat, np.sqrt(self.Vt_hat) + self.epsilon)
        return updated_grad



class simple_NN(object):
    
    #initialize  weight and Bias
    def __init__(self,X,Y,hidden_layer_node_list=[4,3],epoch=100,learning_rate=0.01,batch_size=1000):
        #Initialize synapses weight abd Bias
        self.W = []
        self.B = []
        self.X = X
        self.Y = Y
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        for i in range(len(hidden_layer_node_list)):
            if i==0:
                self.W.append(np.random.uniform(-1,1,size=(X.shape[1],hidden_layer_node_list[i])))
            else:
                self.W.append(np.random.uniform(-1,1,size=(hidden_layer_node_list[i-1],hidden_layer_node_list[i])))
            self.B.append(np.random.random((1,hidden_layer_node_list[i])))
        #Appeend ooutput layer weight and Bias
        self.W.append(np.random.uniform(-1,1,size=(hidden_layer_node_list[-1],Y.shape[1])))
        self.B.append(np.random.random((1,Y.shape[1])))
        
        #Initial Adam Descent parameter buffer
        self.Adam_W=[AdamOptimizer(self.learning_rate) for i in range(len(self.W))]
        self.Adam_B=[AdamOptimizer(self.learning_rate) for i in range(len(self.B))]
        
    
    def feed_forward(self,input_layer):
         #Append Input layer
        L=[input_layer]
        num_layer=len(self.W)
        #Append Hidden layer
        for i in range(num_layer-1):
            L.append(Sigmoid(np.dot(L[-1],self.W[i])+self.B[i]))
        #Append output layer
        L.append(softmax(np.dot(L[-1],self.W[-1])+self.B[-1]))    
        return L
        
    
    #feed backward and Update weights and Bias     
    def back_propagation(self,L,Output_error):
        num_layer=len(self.W)
        L_delta=[Output_error *Sigmoid(L[-1],deriv=True)]
        for i in reversed(range(1,num_layer)):
            temp_error=L_delta[0].dot(self.W[i].T)
            temp_delta=temp_error*Sigmoid(L[i],deriv=True)
            L_delta.insert(0,temp_delta)
            
        #Update weights and Bias
        for i in range(num_layer):
            grad_W= L[i].T.dot(L_delta[i])
            grad_W= self.Adam_W[i].update(grad_W)
            
            grad_B= L_delta[i].sum(axis=0)
            grad_B= self.Adam_B[i].update(grad_B)
            
            self.W[i]-=self.learning_rate* grad_W
            self.B[i]-=self.learning_rate* grad_B
    
        
    def train(self):
        for j in range(self.epoch):
            error=[]
            for mb in minibatcher(self.X, self.Y, self.batch_size): 
                L=self.feed_forward(mb[0])
                Output_error=L[-1]-mb[1]
                error.append(mse(L[-1],mb[1]))
                #update weights and Bias
                self.back_propagation(L,Output_error)
            print('Epoch '+str(j)+' Loss function score: '+str(np.mean(error))+
                  ' Accuracy is:'+str(self.compute_acc(self.X,self.Y)))
            
        
    
    #Compute accuracy
    def compute_acc(self,x,y):
       pred=self.feed_forward(x)[-1]
       pred=np.argmax(pred,1)
       gt=np.argmax(y,1)
       acc=np.mean(pred==gt)
       return acc   
            

if __name__=='__main__':
 
    #load  data Frame  
    df=pd.read_csv('/Users/liunanbo/Desktop/MSEUMAP_3.csv')
    # input data
    x=df.filter(regex='pixel').values
    #output data
    gt=df['GT_Label'].values
    #one-hot coding
    y=[]
    for i in gt:
        temp=np.array([0]*10)
        temp[int(i)]=1
        y.append(temp)
    y=np.array(y)


    #Train NN
    NN=simple_NN(x,y,[64,32,20],100,0.003,batch_size=300)
    NN.train()
    
    print(NN.compute_acc(x,y))
    
    
    
    
    
    
