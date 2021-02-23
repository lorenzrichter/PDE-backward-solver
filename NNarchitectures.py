import torch as pt


class DenseNet(pt.nn.Module):
    def __init__(self, d_in, d_out, lr, problem, arch=[30, 30, 30, 30], seed=42):
        super(DenseNet_2, self).__init__()
        pt.manual_seed(seed)
        self.nn_dims = [d_in] + arch + [d_out]
        self.layers = pt.nn.ModuleList([pt.nn.Linear(sum(self.nn_dims[:i + 1]), self.nn_dims[i + 1])
                                        for i in range(len(self.nn_dims) - 1)])   
        self.optim = pt.optim.Adam(self.parameters(), lr=lr)
        #self.optim = pt.optim.SGD(self.parameters(), lr=lr)
        self.relu = pt.nn.ReLU()
        #pt.nn.functional.tanh

    def forward(self, x):
        for i in range(len(self.nn_dims) - 1):
            if i == len(self.nn_dims) - 2:
                x = self.layers[i](x)
            else:
                x = pt.cat([x, pt.nn.functional.tanh(self.layers[i](x))], dim=1)
        return x


class DenseNet_g(pt.nn.Module):
    def __init__(self, d_in, d_out, lr, problem, arch=[30, 30], seed=42, g_parameter_learnable=False):
        super(DenseNet_g, self).__init__()
        pt.manual_seed(seed)
        self.nn_dims = [d_in] + arch + [d_out]
        self.g_parameter_learnable = g_parameter_learnable
        self.layers = pt.nn.ModuleList([pt.nn.Linear(sum(self.nn_dims[:i + 1]), self.nn_dims[i + 1])
                                        for i in range(len(self.nn_dims) - 1)])  
        if self.g_parameter_learnable:
        	self.alpha = pt.nn.Parameter(pt.tensor([1.0]), requires_grad=True)
        	self.register_parameter('alpha', self.alpha)

        self.optim = pt.optim.Adam(self.parameters(), lr=lr)
        #self.optim = pt.optim.SGD(self.parameters(), lr=lr)
        self.relu = pt.nn.ReLU()
        self.g = problem.g
        #pt.nn.functional.tanh

    def forward(self, x):
        g_ = self.g(x).unsqueeze(1)
        for i in range(len(self.nn_dims) - 1):
            if i == len(self.nn_dims) - 2:
                x = self.layers[i](x)
            else:
                x = pt.cat([x, pt.nn.functional.tanh(self.layers[i](x))], dim=1)
        if self.g_parameter_learnable:
        	return x + self.alpha * g_
        return x + g_
