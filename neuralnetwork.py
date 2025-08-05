import numpy as np
def f(x,y):
    return x**2+y**2 
def gradient(f,x,y,h=1e-5):
    return np.array([(f(x+h,y)-f(x,y))/h,(f(x,y+h)-f(x,y))/h])
    

def gradientdescent(f,init,lr=0.1,steps=20):
    x=np.array(init,dtype=float)
    for i in range(steps):
        a=gradient(f,*x)
        x-=lr*a 
gradientdescent(f,np.array([5.0,5.0]))  

def sgd(weights,bias,dW,dB,learningrate=0.1):
    weights-=learningrate*dW
    bias-=learningrate*dB
def relu(f):
    return np.maximum(0,f)

def sigmoid(x):
    return 1/(1+np.exp(-x))

class MSELoss:
    def forward(self,y_pred,y_true):
        return np.mean((y_pred-y_true)**2)

    def backward(self,y_pred,y_true):
        n=y_true.shape[0]
        inigradient=(2/n)*(y_pred-y_true)
        return inigradient

def relu_derivative(x):
    return np.where(x>0,1,0)

   
def sigmoid_derivative(x):
    return (sigmoid(x)*(1-(sigmoid(x))))


class Layer():
    def __init__(self,inputs,neurons,activation,activation_derivative):
        self.weights=np.random.randn(inputs,neurons)*0.01
        self.bias=np.zeros((1,neurons))
        self.afn=activation
        self.afnd=activation_derivative

    def forward(self,inputs):
        self.inputs=inputs
        self.Z=np.dot(inputs,self.weights)+self.bias
        self.A=self.afn(self.Z)
        return self.A
    
    def backward(self,d_output):
        dZ=d_output*self.afnd(self.Z)
        self.dW=np.dot(self.inputs.T,dZ)
        self.dB=np.sum(dZ,axis=0,keepdims=True)
        d_inputs=np.dot(dZ,self.weights.T)
        return d_inputs

class NeuralNetwork():
    def __init__(self):
        self.layers=[]

    def add(self,layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x=layer.forward(x)
        return x    

    def backward(self,d_loss):
        for layer in reversed(self.layers):
            d_loss=layer.backward(d_loss)

    def update(self):
        for layer in self.layers:
            sgd(layer.weights,layer.bias,layer.dW,layer.dB)

    def train(self,x,y_true,loss_fn,epochs,learning_rate=0.1,v=True):
        for epoch in range(epochs):
            y_pred=self.forward(x)
            loss=loss_fn.forward(y_pred,y_true)
            d_loss=loss_fn.backward(y_pred,y_true)
            self.backward(d_loss)
            self.update()
            if v:
                print(f"Epoch {epoch+1}/{epochs},Loss: {loss:.3f}")

    def predict(self,x):
        return self.forward(x)
    
def create_dataset(n_samples=500):
    X, y = [], []
    for i in range(n_samples):
        if i % 2 == 0:
            radius = 4.0
            label = 1
        else:
            radius = 2.0
            label = 0
            
        angle = i * 3.14159 * 2 / n_samples
        x1 = radius * np.cos(angle) + np.random.randn() * 0.2
        x2 = radius * np.sin(angle) + np.random.randn() * 0.2
        X.append([x1, x2])
        y.append([label])
        
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(n_samples=1000)

nn = NeuralNetwork()
loss_function = MSELoss()

nn.add(Layer(inputs=2, neurons=16, activation=relu, activation_derivative=relu_derivative))
nn.add(Layer(inputs=16, neurons=1, activation=sigmoid, activation_derivative=sigmoid_derivative))
    
nn.train(X_train, y_train, loss_fn=loss_function, epochs=2000, v=True)
test_point_inner = np.array([[0, 0]])
test_point_outer = np.array([[4.0, 0]])
    
pred_inner = nn.predict(test_point_inner)
pred_outer = nn.predict(test_point_outer)
    
print(f"Prediction for point {test_point_inner}: {pred_inner[0][0]:.4f}")
print(f"Prediction for point {test_point_outer}: {pred_outer[0][0]:.4f}")


       

    









