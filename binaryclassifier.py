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

class BCELoss:
    def forward(self,y_pred,y_true):
        eps=1e-8
        return  -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
    def backward(self,y_pred,y_true):
        return y_pred-y_true

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
        self.y_pred=sigmoid(self.Z)
        return self.y_pred
    
    def backward(self,d_output):
        dZ=d_output*self.afnd(self.Z)
        self.dW=np.dot(self.inputs.T,dZ)
        self.dB=np.sum(dZ,axis=0,keepdims=True)
        d_inputs=np.dot(dZ,self.weights.T)
        return d_inputs

class BinaryClassifier():
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
    
def create_moons_dataset(n_samples=500, noise=0.1):
    n_samples_per_moon = n_samples // 2
    
    
    X0 = np.linspace(0, np.pi, n_samples_per_moon)
    x1_0 = np.cos(X0)
    x2_0 = np.sin(X0)
    
    X1 = np.linspace(0, np.pi, n_samples_per_moon)
    x1_1 = 1 - np.cos(X1)
    x2_1 = 0.5 - np.sin(X1)
    
    X = np.vstack([
        np.stack([x1_0, x2_0], axis=1),
        np.stack([x1_1, x2_1], axis=1)
    ])
    
    y = np.array([0] * n_samples_per_moon + [1] * n_samples_per_moon).reshape(-1, 1)
    
    X += np.random.randn(*X.shape) * noise
    
    return X, y

x_train,y_train=create_moons_dataset()
classifier=BinaryClassifier()
loss=BCELoss()
classifier.add(Layer(inputs=2,neurons=16,activation=sigmoid,activation_derivative=sigmoid_derivative))
classifier.add(Layer(inputs=16,neurons=1,activation=sigmoid,activation_derivative=sigmoid_derivative))

classifier.train(x_train,y_train,loss,2000,v=True)

test_point_top_moon = np.array([[0, 0.5]])  
test_point_bottom_moon = np.array([[1, 0]])  

pred_top = classifier.predict(test_point_top_moon)
pred_bottom = classifier.predict(test_point_bottom_moon)

print(f"\nPrediction for point in top moon (class 0) {test_point_top_moon}: {pred_top[0][0]:.4f}")
print(f"Prediction for point in bottom moon (class 1) {test_point_bottom_moon}: {pred_bottom[0][0]:.4f}")
