import torch
from torch.optim.optimizer import Optimizer
class ExtraGradient(Optimizer):
    MONOTONE = True
    SKIP_TEST_LOGREG = True
    def __init__(self,params, L: float = 1., last_iterate: bool = False,
                 verbose: bool = True, testing: bool = False):
        super().__init__(params, dict(L=L))

        self.verbose = verbose
        self.testing = testing
        self.last_iterate = last_iterate
        if len(self.param_groups) != 1:
            raise ValueError("ExtraGradient doesn't support per-parameter options "
                             "(parameter groups)")
        if not self.last_iterate:
            group = self.param_groups[0]
            params = group['params']
            p = next(iter(params))
            state_common = self.state[p]
            state_common['lambda_sum'] = 0.

        self.lr = 1./L
        # Initialization of intermediate points
        for p in params:
            state = self.state[p]
            state['x'] = p.detach().clone()
            if not self.last_iterate:
                state['x_average'] = torch.zeros_like(p)

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): a closure that reevaluates the model and returns the loss.
        """
        closure = torch.enable_grad()(closure)

        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        params = group['params']

        if not self.last_iterate:
            p = next(iter(params))
            state_common = self.state[p]
            with torch.no_grad():
                for p in params:
                    state = self.state[p]
                    p.zero_().add_(state['x'])

        operator = closure()
        with torch.no_grad():
            for p, g in zip(params, operator):
                p.sub_(g, alpha = self.lr)
        operator_2 = closure()
        with torch.no_grad():
            for p, g in zip(params,operator_2):
                state = self.state[p]
                state['x'].sub_(g, alpha = self.lr)

                if not self.last_iterate:
                    state['x_average'].mul_(state_common['lambda_sum']).add_(self.lr * state['x'])
                    state_common['lambda_sum'] += self.lr
                    state['x_average'].div_(state_common['lambda_sum'])
                    p.zero_().add_(state['x_average'])
                else:
                    p.zero_().add_(state['x'])
