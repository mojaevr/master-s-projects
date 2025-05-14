from torch.optim.optimizer import Optimizer
import torch
from utils import tuple_to_vector, rollup_vector
EPS = 1e-6


class VIQA(Optimizer):
    def __init__(self,
                 params,
                 L: float = 1., delta: float = 0., B0: float = 0., memory: int = 10, p_order: int = 2,
                 last_iterate: bool = True,
                 verbose: bool = True, testing: bool = False, qn='Broyden_damped', half=1
                 ):
        super().__init__(params, dict(L=L))

        self.verbose = verbose
        self.testing = testing
        self.qn = qn
        self.half = half

        self.last_iterate = last_iterate
        self.p_order = p_order
        self.L = L
        self.delta = delta
        self.B0 = B0
        self.memory = memory

        if len(self.param_groups) != 1:
            raise ValueError("VIQA doesn't support per-parameter options "
                             "(parameter groups)")
        group = self.param_groups[0]
        params = group['params']
        p = next(iter(params))
        state_common = self.state[p]
        state_common['k'] = 0
        state_common['S_qn'] = []
        state_common['Y_qn'] = []

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
    def compute_lambda(self, L, params, delta=0.):
        bound = (1 / (20 * self.p_order - 8) + 1 / (10 * self.p_order + 2)) / 2
        den = 1
        for m in range(1, self.p_order + 1):
            den *= m
        norm = 0.
        with torch.no_grad():
            for p in params:
                state = self.state[p]
                norm += (p - state['v']).square().sum() ** .5
        return bound / (L * norm ** (self.p_order - 1) / den + delta)

    @torch.no_grad()
    def full_inverse_vector(self, A, b, tau):
        return torch.linalg.inv(A + torch.diag(torch.ones_like(b)).mul_(tau)) @ b

    @torch.no_grad()
    def subproblem_solver_qn(self, U, V, C, F, B0, L, delta=0., tau_up=0.01, tau_low=0., max_iter=20, testing=False):
        if len(U) == 0:
            U = torch.zeros_like(F)
            V = torch.zeros_like(F)
            C = torch.tensor([1.])
            UT = torch.zeros_like(F)
            VUT = UT.mul(UT).sum()
            VF = torch.zeros_like(F)
            C_inv = torch.zeros_like(F)

            def woodbury_inverse_vector(UT, VUT, VF, C_inv, F, chi):
                return F / chi
        else:
            UT = U.T
            VUT = V @ UT
            VF = V @ F
            C_inv = torch.reciprocal(C)

            def woodbury_inverse_vector(UT, VUT, VF, C_inv, F, chi):
                inverse_vf = torch.linalg.inv(torch.diag(C_inv) + VUT / chi) @ VF
                return F / chi - UT @ inverse_vf / chi ** 2
            # def woodbury_inverse_vector(UT, VUT, VF, C_inv, F, chi):
            #    inverse_vf = torch.linalg.inv(C_inv + VUT/chi) @ VF
            #   return F/chi - UT @ inverse_vf / chi **2

        j = 0
        flag = True
        while flag and j < max_iter:
            h = - woodbury_inverse_vector(UT, VUT, VF, C_inv, F, chi=B0 + 5 * L * tau_up + delta)
            norm = torch.linalg.norm(h).item()
            if norm > tau_up:
                tau_up *= 2
            else:
                flag = False
            j += 1

        j = 0
        h = torch.zeros_like(F)
        norm = 0.
        criteria = 100.
        while j < max_iter and criteria > L / 2 * norm ** 2:  # and abs(tau - norm) > 0.001
            tau = (tau_up + tau_low) / 2
            h = - woodbury_inverse_vector(UT, VUT, VF, C_inv, F, chi=B0 + 5 * L * tau + delta)
            norm = torch.linalg.norm(h).item()
            # print(tau, norm)
            if norm < tau:
                tau_up = tau + 0.5
            else:
                tau_low = tau + 0.5
            j += 1
            Ah = U.mul(C * V @ h) if U.dim() <= 1 else (U.T.mul(C)) @ (V @ h)
            if testing: assert torch.norm(F + Ah + (B0 + 5 * L * tau + delta) * h) < EPS
            c = F + Ah + 5 * L * torch.linalg.norm(h) * h + (B0 + delta) * h
            criteria = torch.norm(c)
        # print(criteria, 'criteria')
        return h  # , criteria

    @torch.no_grad()
    def subproblem_solver(self, A, b, L, inverse_vector, delta=0., tau_up=0.01, tau_low=0., max_iter=20):

        j = 0
        flag = True
        while flag and j < max_iter:
            h = - inverse_vector(A, b, 5 * L * tau_up + delta)
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
            h = - inverse_vector(A, b, 5 * L * tau + delta)
            norm = torch.linalg.norm(h).item()
            # print(tau, norm)
            if norm < tau:
                tau_up = tau + 0.5
            else:
                tau_low = tau + 0.5
            j += 1
            c = b + A @ h + 5 * L * torch.linalg.norm(h) * h + delta * h
            criteria = torch.norm(c)
        return h  # , criteria

    def Broyd_qn(self, params, S, Y, B0=0., damping=0., testing=False):
        V_qn = []
        C_qn = []
        U_qn = []
        g = tuple_to_vector(list(params))
        B = torch.diag(torch.ones_like(g)).mul_(B0)

        if len(S) == 0:
            return V_qn, C_qn, U_qn, B

        for i in range(len(S)):
            s = tuple_to_vector(S[-1 - i])
            v = s.clone()
            y = tuple_to_vector(Y[-1 - i])
            u = y - B @ s
            c = 1 / (v.mul(v).sum() * (damping + 1))
            C_qn.insert(0, c)
            V_qn.insert(0, v)
            U_qn.insert(0, u)
            B += u.outer(v) * c

        V_qn, C_qn, U_qn = torch.stack(V_qn), torch.stack(C_qn), torch.stack(U_qn)
        if testing:
            B_new = torch.diag(torch.ones_like(g).mul(B0)) + U_qn.T.mul(C_qn) @ V_qn
            assert torch.linalg.norm(B - B_new) < EPS
        return V_qn, C_qn, U_qn, B

    def BFGS_qn(self, params, S, Y, B0, damping=0, testing=False):
        V_qn = []
        C_qn = []
        U_qn = []
        g = tuple_to_vector(list(params))
        B = torch.diag(torch.ones_like(g)).mul_(B0)

        if len(S) == 0:
            return V_qn, C_qn, U_qn, B

        for i in range(1, len(S)):
            s = tuple_to_vector(S[-1 - i])
            y = tuple_to_vector(Y[-1 - i])

            # part 1
            c1 = 1 / (y.mul(s).sum() * (damping + 1))
            C_qn.insert(0, c1)
            V_qn.insert(0, s)
            U_qn.insert(0, y)

            # part 2
            u = B @ s
            c2 = - 1 / s.mul(u).sum()
            C_qn.insert(0, c2)
            V_qn.insert(0, u)
            U_qn.insert(0, u)
            B += u.outer(u) * c2
            B += y.outer(y) * c1

        V_qn, C_qn, U_qn = torch.stack(V_qn), torch.stack(C_qn), torch.stack(U_qn)
        if testing:
            B_new = torch.diag(torch.ones_like(g).mul(B0)) + U_qn.T.mul(C_qn) @ V_qn
            assert torch.linalg.norm(B - B_new) < EPS
        return V_qn, C_qn, U_qn, B

    # step 3
    def qn_step(self, params, operator_v, S, Y, B0=0, testing=False):
        b = tuple_to_vector(operator_v).detach().clone()

        if self.qn == 'Broyden_damped':
            V_qn, C_qn, U_qn, B = self.Broyd_qn(params, S, Y, B0, damping=self.memory, testing=self.testing)
        elif self.qn == 'BFGS':
            V_qn, C_qn, U_qn, B = self.BFGS_qn(params, S, Y, B0, damping=0, testing=self.testing)
        elif self.qn == 'BFGS_broid':
            V_qn1, C_qn1, U_qn1, B1 = self.BFGS_qn(params, S, Y, B0, damping=0)
            V_qn2, C_qn2, U_qn2, B2 = self.Broyd_qn(params, S, Y, B0)
            V_qn = torch.stack([V_qn1, V_qn2])
            U_qn = torch.stack([U_qn1, U_qn2])
            C_qn = torch.stack([C_qn1, C_qn2])
            B = (B1 + B2) / (1 + self.half)
        elif self.qn == 'BFGS_damped':
            V_qn1, C_qn1, U_qn1, B1 = self.BFGS_qn(params, S, Y, B0, damping=0)
            V_qn2, C_qn2, U_qn2, B2 = self.Broyd_qn(params, S, Y, B0, damping=0)
            V_qn = torch.stack([V_qn1, V_qn2])
            U_qn = torch.stack([U_qn1, U_qn2])
            C_qn = torch.stack([C_qn1, C_qn2])
            B = (B1 + B2) / (1 + self.half)
        elif self.qn == 'damped_BFGS':
            V_qn, C_qn, U_qn, B = self.BFGS_qn(params, S, Y, B0, damping=self.memory)
        elif self.qn == 'damped_BFGS_broid':
            V_qn1, C_qn1, U_qn1, B1 = self.BFGS_qn(params, S, Y, B0, damping=self.memory)
            V_qn2, C_qn2, U_qn2, B2 = self.Broyd_qn(params, S, Y, B0)
            V_qn = torch.stack([V_qn1, V_qn2])
            U_qn = torch.stack([U_qn1, U_qn2])
            C_qn = torch.stack([C_qn1, C_qn2])
            B = (B1 + B2) / (1 + self.half)
        elif self.qn == 'damped_BFGS_damped':
            V_qn1, C_qn1, U_qn1, B1 = self.BFGS_qn(params, S, Y, B0, damping=self.memory)
            V_qn2, C_qn2, U_qn2, B2 = self.Broyd_qn(params, S, Y, B0, damping=self.memory)
            V_qn = torch.stack([V_qn1, V_qn2])
            U_qn = torch.stack([U_qn1, U_qn2])
            C_qn = torch.stack([C_qn1, C_qn2])
            B = (B1 + B2) / (1 + self.half)
        elif self.qn == 'broyd':
            V_qn, C_qn, U_qn, B = self.Broyd_qn(params, S, Y, B0)
        else:
            raise ValueError
        h = self.subproblem_solver_qn(U=U_qn, V=V_qn, C=C_qn, F=b, B0=B0, L=self.L, delta=self.delta,
                                      testing=self.testing)
        if testing:
            h_direct = self.subproblem_solver(A=B, b=b, L=self.L, inverse_vector=self.full_inverse_vector,
                                              delta=self.delta)
            assert torch.linalg.norm(h - h_direct) < EPS
        h_tuple = rollup_vector(h, list(params))
        return h_tuple

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
                state['v'] = state['x0'] + state['s']
                p.zero_().add_(state['v'])

        operator_v = closure()

        h_tuple = self.qn_step(params, operator_v, S=state_common['S_qn'], Y=state_common['Y_qn'], B0=self.B0,
                               testing=self.testing)

        state_common['S_qn'].insert(0, h_tuple)
        if len(state_common['S_qn']) > self.memory:
            state_common['S_qn'].pop()

        with torch.no_grad():
            for p, z in zip(params, h_tuple):
                state = self.state[p]
                p.add_(z)
                state['x'] = p.clone().detach()

        with torch.no_grad():
            for p in params:
                state = self.state[p]
                state['x'] = p.clone()

        lamb = self.compute_lambda(self.L, params, delta=self.delta)
        operator_x = closure()

        oper_dif = []
        with torch.no_grad():
            for g2, g1 in zip(operator_x, operator_v):
                oper_dif.append(g2 - g1)

        state_common['Y_qn'].insert(0, oper_dif)
        if len(state_common['Y_qn']) > self.memory:
            state_common['Y_qn'].pop()

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
