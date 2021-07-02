
class Optimizer(object):
    def __init__(self,lr):
        self.lr= lr
    def __call__(self,X,Y,model):
        pass
    def update_weights(self,W,gradW):
        pass

class BGD(Optimizer):
    def __init__(self,lr,bs,lambd):
        super().__init__(lr)
        self.bs=bs
        self.lambd=lambd

    def __call__(self,X,grad_loss,model):
        model.backward(X,grad_loss)

    def update_weights(self,W,gradW):
        W-=(self.lr*(gradW+ self.lambd*W))
