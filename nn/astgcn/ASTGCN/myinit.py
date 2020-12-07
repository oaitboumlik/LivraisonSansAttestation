import mxnet as mx

class MyInit(mx.init.Initializer):
    xavier = mx.init.Xavier()
    uniform = mx.init.Uniform()

    def _init_weight(self, name, data):
        if len(data.shape) < 2:
            self.uniform._init_weight(name, data)
            print('Init', name, data.shape, 'with Uniform')
        else:
            self.xavier._init_weight(name, data)
            print('Init', name, data.shape, 'with Xavier')
