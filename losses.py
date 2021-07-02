import numpy as np

class Loss():
    def __call__(self,s,y,flag='Problema_XOR'):
        pass
    def gradient(self,s,y,flag='Problema_XOR'):
        pass

class MSE(Loss):
    def __call__(self,scores,y_true,flag='Problema_XOR'):
        if flag=='Problema_XOR':
            return np.mean(np.sum((scores - y_true) ** 2, axis=1))
        else:
            yy_true=np.zeros(scores.shape)
            yy_true[np.arange(scores.shape[0]),y_true]=1
            return np.mean(np.sum((scores - yy_true) ** 2, axis=1))

    def gradient(self,scores,y_true, flag='Problema_XOR'):
        if flag == 'Problema_XOR':
            return 2*(scores-y_true)
        else:
            yy_true = np.zeros(scores.shape)
            yy_true[np.arange(scores.shape[0]), y_true] = 1
            return 2*(scores-yy_true)

class CCE(Loss):
    def __call__(self,score,y_true,flag='Problema_XOR'):
        score -= np.max(score, axis=1)[:, np.newaxis]
        score_real = score[np.arange(len(score)), y_true]
        e_score = np.exp(score)
        e_score_sum = np.sum(e_score, axis=1)
        s = np.log(e_score_sum) - score_real
        return np.mean(s)

    def gradient(self,score,y_true,flag='Problema_XOR'):
        score -= np.max(score, axis=1)[:, np.newaxis]
        e_score = np.exp(score)
        e_score_sum = np.sum(e_score, axis=1)
        g = np.zeros(score.shape)
        g[np.arange(len(score)), y_true] = -1
        grad = g + (e_score / e_score_sum[:, np.newaxis])
        return grad
