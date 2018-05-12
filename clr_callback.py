from keras.callbacks import *
import keras.backend as K
import numpy as np

class CyclicLR(Callback):
    """
    This callback implements a cyclic learning rate policy (CLR).
    This class has three built in policies:
    `triangular`:
        A basic triangular cycle with no amplitude scaling.
    `triangular2`:
        A basic triangular cycle that scale initial amplitude by half each cycle.
    `exp_range`:
        A cycle that scales initial aplitude by gamma**(cycle iteration) at each cycle iteration.
    ------
    Parameters:
        base_lr: initial learning rate which is the lower boundary in the cycle.
        max_lr: upper boudary in the cycle.
        step_size: number of training iteration per half cycle. Suggested step_size is
        2-8 x training iterations in epoch.
        mode: one of {`triangular`, `triangular2`, `exp_range`}.
            Default `triangular`.
        gamma: constant in `exp_range` scaling function
        scale_fn: Custom scaling policy defined by a single argument lambda function,
            where 0 <= scale_fn(x) <= 1 for all x >= 0
            mode parameter is ignored
        scale_mode: {`cycle`, `iteration`}.
            Defines whether scale_fn is evaluated on cycle number or cycle iterations
            (training iterations since start of cycle). Default is `cycle`.
    """

    def __init__(self, base_lr = 0.001, max_lr=0.006, step_size=2000., mode='triangular', gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iteration'
        else:
            self.scale_fn = scale_fn
            self.scale_fn = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None, new_step_size=None):
        """
        Reset cycle iterations.
        Optimal boundary/step_size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0. 

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr + self.base_lr) * np.maximum(0, (1-x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr + self.base_lr) * np.maximum(0, (1-x)) * self.scale_fn(self.clr_iterations)

    def on_train_begin(self, logs={}):
    
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())
    
    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.mode.optimizer.lr)) 
        self.history.setdefault('iterations', []).append(self.trn_iterations)
        
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())