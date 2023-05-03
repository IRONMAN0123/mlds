import numpy as np
x=np.array(([2,9],[1,5],[3,6]),dtype=float)
print (x)
y=np.array(([92],[86],[89]),dtype=float)
#Normalization of dataset
x=x/np.max(x,axis=0)
y=y/100
#print(x)
#print(y)
def sigmoid(x):
 return (1/(1+np.exp(-x)))
def derivatives_sigmoid(x):
 return x*(1-x)
epoch=1000
lr=0.01
input_layer_neurons=2
hidden_layer_neurons=2
output_neurons=1
wh=np.random.uniform(size=(input_layer_neurons,hidden_layer_neurons))
#print(wh)
bh=np.random.uniform(size=(1,hidden_layer_neurons))
#print(bh)
wout=np.random.uniform(size=(hidden_layer_neurons,output_neurons))
#print(wout)
bout=np.random.uniform(size=(1,output_neurons))
#print(bout)
for i in range(epoch):
 hinp1=np.dot(x,wh)
 hinp=hinp1+bh
 hlayer_act=sigmoid(hinp)
 ##print(hlayer_act)
 outinp1=np.dot(hlayer_act,wout)
 outinp=outinp1+bout
 output=sigmoid(outinp)
 
 #print(EO)
# back Propagation
 EO=(y-output)
 outgrad=derivatives_sigmoid(output)
 d_output=EO*outgrad
 EH=d_output.dot(wout.T)
 #print(EH)
 hiddengrad=derivatives_sigmoid(hlayer_act)
 #print(hiddengrad)
 d_hiddenlayer=EH*hiddengrad
 #print(d_hiddenlayer)
 #change of weight at each layer
 wout+=hlayer_act.T.dot(d_output)*lr
 #print(wout)
 bout += np.sum(d_output,axis=0,keepdims=True) *lr
 #print(bout)
 wh+=x.T.dot(d_hiddenlayer)*lr
 bh +=np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr
 #output after each epoch
 print ("-----------Epoch-", i+1, "Starts----------")
 print("Input: \n" + str(x))
 print("Actual Output: \n" + str(y))
 print("Predicted Output: \n" ,output)
 print("Error:\n"+str(EO))
 print("-----------Epoch-", i+1, "Ends----------\n")
print("Actual ouput"+str(y))
print("Predicted Output"+str(output))
print("Error"+str(EO))
