import torch
import time
from tqdm.notebook import tqdm as tqdm
# return a flat vector from a tuple of vectors and matrices.
def tuple_to_vector(tuple_in):
    return torch.cat([t.view(-1) for t in tuple_in])

def tuple_numel(tuple_in):
    return [t.numel() for t in tuple_in]

# return a tuple of vectors from a flat vec
def rollup_vector(flat_vector, tuple_in):
    new_vec = torch.split(flat_vector, tuple_numel(tuple_in))
    return [v.view_as(t) for v, t in zip(new_vec, tuple_in)]

def fit_vi(optimizer, iters_num, F, param, gap, precision:float = 1e-50, **kwargs):
    gaps = []
    grads = []
    times = [0.]
    for iters in tqdm(range(iters_num)):
        def closure():
            optimizer.zero_grad()
            return F(param)
        gr = closure()
        norm_grad_sq = 0.
        for g in gr:
            norm_grad_sq += g.square().sum()
        grads.append(norm_grad_sq.item())
        func_loss = gap(param)
        gaps.append(func_loss.item())
        st = time.time()
        optimizer.step(closure)
        timed = time.time()-st
        times.append(times[-1] + timed)
    return gaps, grads, times