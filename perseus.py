import torch
from torch.optim.optimizer import Optimizer
from utils import tuple_to_vector, rollup_vector


class Perseus(Optimizer):
    MONOTONE = False
    SKIP_TEST_LOGREG = True

    def __init__(self,
                 params,
                 L: float = 1., p_order: int = 2, last_iterate: bool = True,
                 verbose: bool = True, testing: bool = False
                 ):
        super().__init__(params, dict(L=L))

        self.verbose = verbose
        self.testing = testing

        self.last_iterate = last_iterate
        self.p_order = p_order
        self.L = L

        if len(self.param_groups) != 1:
            raise ValueError("Perseus doesn't support per-parameter options "
                             "(parameter groups)")
        group = self.param_groups[0]
        params = group['params']
        p = next(iter(params))
        state_common = self.state[p]
        state_common['k'] = 0

        if not self.last_iterate:
            state_common['lambda_sum'] = 0.

        # Initialization of intermediate points
        for p in params:
            state = self.state[p]
            state['x0'] = p.detach().clone()
            state['x'] = state['x0'].clone()
            state['v'] = state['x0'].clone()
            state['s'] = torch.zeros_like(state['x'])
            if not self.last_iterate:
                state['x_average'] = torch.zeros_like(p)

    # lambda_computation
    @torch.no_grad()
    def compute_lambda(self, L, params):
        bound = (1 / (20 * self.p_order - 8) + 1 / (10 * self.p_order + 2)) / 2
        den = 1
        for m in range(1, self.p_order + 1):
            den *= m
        norm = 0.
        with torch.no_grad():
            for p in params:
                state = self.state[p]
                norm += (p - state['v']).square().sum()
        return bound * den / (L * norm ** (self.p_order - 1))

    @torch.no_grad()
    def full_inverse_vector(self, A, b, tau):
        return torch.linalg.inv(A + torch.diag(torch.ones_like(b)).mul_(tau)) @ b

    @torch.no_grad()
    def subproblem_solver(self, A, b, L, inverse_vector, tau_up=0.01, tau_low=0., max_iter=20):

        j = 0
        flag = True
        while flag and j < max_iter:
            h = - inverse_vector(A, b, 5 * L * tau_up)
            norm = torch.linalg.norm(h).item()
            if norm > tau_up:
                tau_up *= 2
            else:
                flag = False
            j += 1

        j = 0
        h = torch.zeros_like(b)
        norm = 0.
        criteria = 100.
        while j < max_iter and criteria > L / 2 * norm ** 2:  # and abs(tau - norm) > 0.001
            tau = (tau_up + tau_low) / 2
            h = - inverse_vector(A, b, 5 * L * tau)
            norm = torch.linalg.norm(h).item()
            # print(tau, norm)
            if norm < tau:
                tau_up = tau + 0.
            else:
                tau_low = tau + 0.
            j += 1
            c = b + A @ h + 5 * L * torch.linalg.norm(h) * h
            criteria = torch.norm(c)
        return h  # , criteria

    # step 3
    def second_order_step(self, params, operator_v):
        operator_vec = tuple_to_vector(operator_v)
        A = self.jacobian(params, operator_vec)
        b = operator_vec.detach().clone()
        h = self.subproblem_solver(A, b, L=self.L, inverse_vector=self.full_inverse_vector)
        h_tuple = rollup_vector(h, list(params))
        with torch.no_grad():
            for p, z in zip(params, h_tuple):
                p.add_(z)

    def jacobian(self, params, operator_vector):
        full_jacobian = []
        for g in operator_vector:
            temp_jvp = torch.autograd.grad(g, params, retain_graph=True)
            full_jacobian.append(tuple_to_vector(temp_jvp))
        return torch.stack(full_jacobian)

    def step(self, closure):

        closure = torch.enable_grad()(closure)

        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        params = group['params']
        p = next(iter(params))
        state_common = self.state[p]

        # step 2
        with torch.no_grad():
            for p in params:
                state = self.state[p]
                state['v'] = state['x0'] + state['s']  # / self.L
                p.zero_().add_(state['v'])

        operator_v = closure()
        if self.p_order == 1:
            with torch.no_grad():
                for p, g in zip(params, operator_v):
                    state = self.state[p]
                    p.sub_(g / self.L / 5)
                    state['x'] = p.clone()
        elif self.p_order == 2:
            self.second_order_step(params, operator_v)
            with torch.no_grad():
                for p in params:
                    state = self.state[p]
                    state['x'] = p.clone()

        lamb = self.compute_lambda(self.L, params)
        operator_x = closure()

        if not self.last_iterate:
            state_common['lambda_sum'] += lamb

        with torch.no_grad():
            for p, g in zip(params, operator_x):
                state = self.state[p]
                state['s'].sub_(g, alpha=lamb)

                if not self.last_iterate:
                    state['x_average'].mul_(state_common['lambda_sum']).add_(state['x'] * lamb)
                    state_common['lambda_sum'] += lamb
                    state['x_average'].div_(state_common['lambda_sum'])
                    p.zero_().add_(state['x_average'])

        state_common['k'] += 1