import numpy as np

class BaseLayer():
    def __init__(self):
        pass
    def get_output_shape(self):
        pass
    def set_output_shape(self):
        pass

class Input(BaseLayer):
    def set_output_shape(self, x):
        self.output_shape = x.shape

    def get_output_shape(self,x):
        return self.output_shape


class ConcatInput(BaseLayer):
    def __call__(self,x,y):
        self.input1_shape=x.shape
        self.input2_shape=y.shape
        concat=np.hstack((x,y))
        self.output_shape=concat.shape
        return concat

    def get_input1_shape(self):
        return self.input1_shape

    def get_input2_shape(self):
        return self.input2_shape

    def get_output_shape(self):
        return self.output_shape

    def set_output_shape(self,shape):
        self.output_shape=[shape[0],shape[1]]

    def build_bias(self,s):
        x_aux = np.hstack((np.ones((len(s), 1)), s))
        return x_aux


class WLayer(BaseLayer):
    def __init__(self,n_neuronas,act,xdim,bs):
        super().__init__()
        self.n_neurons=n_neuronas
        self.activation=act
        self.xdim=xdim
        self.input_shape=[bs,xdim+1]
        self.output_shape=[bs,self.n_neurons]
        self.W=np.zeros((xdim+1,self.n_neurons))

    def get_input_shape(self):
        return self.input_shape

    def set_input_shape(self,shape):
        self.input_shape=[shape[0],shape[1]]

    def set_output_shape(self, shape):
        self.output_shape= [shape[0],shape[1]]

    def get_output_shape(self):
        return self.output_shape

    def get_weights(self):
        return self.W

    def update_weights(self,opt,gradW):
        opt.update_weights(self.W ,gradW)

class Dense(WLayer):
    def init_weights(self):
       # self.W +=np.random.normal(0,0.5,size=(self.xdim+1,self.n_neurons))
        self.W +=np.random.randn(self.xdim+1,self.n_neurons) * 2 / np.sqrt(self.xdim+1+self.n_neurons)
        self.W[0]=0

    def build_bias(self,s):
        x_aux = np.hstack((np.ones((len(s), 1)), s))
        return x_aux

    def producto(self,x):
        xt=self.build_bias(x)
        return xt.dot(self.W)

    def __call__(self,x):
        y=self.producto(x)
        return self.activation(y)























