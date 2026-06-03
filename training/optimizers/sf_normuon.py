import math
import torch
import torch.distributed as dist
from .polar import polar_fast


@torch.compile
def _compute_ip_term(grad, z, x, sf_beta1_k):
    """Fuses inner product subtraction and reduction to eliminate allocations."""
    return sf_beta1_k * torch.sum(grad * (z - x))


@torch.compile
def _update_vt_and_get_denom(Pt, vt, beta2, eps):
    """Fuses vt EMA update, sqrt, and eps addition into a single kernel."""
    Pt_sq = Pt.pow(2)
    meancols = Pt_sq.mean(dim=1, keepdim=True)
    vt.mul_(beta2).add_(meancols, alpha=1 - beta2)
    return vt.sqrt() + eps


@torch.compile
def _update_1d_param(z, y, grad, exp_avg, exp_avg_sq, beta1, beta2, bias_correction1, bias_correction2, group_lr, decay, eps):
    """Fuses the entire AdamC 1D update into a single kernel with zero allocations."""
    if decay > 0:
        z.add_(y, alpha=-decay * group_lr * group_lr)
    
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    
    denom = (exp_avg_sq / bias_correction2).sqrt_().add_(eps)
    step = (exp_avg / bias_correction1) * group_lr / denom
    z.sub_(step)


class SFNorMuon(torch.optim.Optimizer):
    """
    SF-NorMuon: Schedule-Free Spectral Optimizer
    =============================================
    
    Combines:
    1. SF-NorMuon for 2D matrix parameters (Polar Express 8, row-wise EMA normalization).
    2. AdamC Polyak for 1D non-matrix parameters (Inverse-L1 gradient norm scaling).
    3. ScheduleFree+ dynamics (Beta annealing, r=1 weighting, c_warmup).
    
    Based on "Anytime Training with Schedule-Free Spectral Optimization"
    and "ScheduleFree+: Scaling Learning-Rate-Free & Schedule-Free Learning".
    """

    def __init__(self, params, lr=1.0, betas=(0.9, 0.95), sf_beta1=0.9, eps=1e-8,
                 weight_decay=0.0, r=1.0, polyak_beta=0.0, c_warmup=0,
                 sf_beta1_anneal_steps=0, sf_beta1_max=0.965, weight_lr_power=2.0,
                 eta_scale=0.2):
        defaults = dict(lr=lr, betas=betas, sf_beta1=sf_beta1, eps=eps, r=r,
                        k=0, train_mode=True, weight_sum=0.0, lr_max=eps,
                        scheduled_lr=0.0, polyak_beta=polyak_beta,
                        sf_beta1_anneal_steps=sf_beta1_anneal_steps,
                        sf_beta1_max=sf_beta1_max, grad_l1_ema=0.0,
                        c_warmup=c_warmup, weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay, eta_scale=eta_scale)
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            if group['train_mode']:
                for p in group['params']:
                    state = self.state[p]
                    if 'x' in state:
                        p.detach().copy_(state['x'])
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            if not group['train_mode']:
                for p in group['params']:
                    state = self.state[p]
                    if 'y' in state:
                        p.detach().copy_(state['y'])
                group['train_mode'] = True

    @torch.no_grad()
    def step(self, closure=None):
        if not self.param_groups[0]['train_mode']:
            raise Exception("Optimizer must be in train mode.")
        
        device = self.param_groups[0]['params'][0].device
        function_value = None
        if closure is not None:
            function_value = closure()
        if function_value is None:
            function_value = torch.tensor(0.0, device=device)
        elif not isinstance(function_value, torch.Tensor):
            function_value = torch.tensor(float(function_value), device=device)
            
        group0 = self.param_groups[0]
        grad_l1_ema = group0['grad_l1_ema']
        k = group0['k']
        polyak_beta = group0['polyak_beta']
        sf_beta1 = group0['sf_beta1']
        sf_beta1_max = group0['sf_beta1_max']
        sf_beta1_anneal_steps = group0['sf_beta1_anneal_steps']
        
        if sf_beta1_anneal_steps > 0:
            progress = min(k / sf_beta1_anneal_steps, 1.0)
            sf_beta1_k = 1 - math.exp(math.log(1 - sf_beta1) * (1 - progress) + math.log(1 - sf_beta1_max) * progress)
        else:
            sf_beta1_k = sf_beta1

        grad_l1_list = []
        ip_term_list = []
        is_distributed = False
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                
                if hasattr(grad, 'to_local'):
                    is_distributed = True
                    local_grad = grad.to_local()
                else:
                    local_grad = grad
                    
                grad_l1_list.append(torch.linalg.vector_norm(local_grad, ord=1))
                state = self.state[p]
                if 'z' in state:
                    z = state['z']
                    x = state['x']
                    local_z = z.to_local() if hasattr(z, 'to_local') else z
                    local_x = x.to_local() if hasattr(x, 'to_local') else x
                    ip_term_list.append(_compute_ip_term(local_grad, local_z, local_x, sf_beta1_k))
                    
        grad_l1 = torch.stack(grad_l1_list).sum() if grad_l1_list else torch.tensor(0.0, device=device)
        ip_term = torch.stack(ip_term_list).sum() if ip_term_list else torch.tensor(0.0, device=device)
        
        if is_distributed and dist.is_available() and dist.is_initialized():
            dist.all_reduce(grad_l1, op=dist.ReduceOp.SUM)
            dist.all_reduce(ip_term, op=dist.ReduceOp.SUM)
            dist_tensor = torch.zeros(1, device=device)
            dist_tensor[0] = function_value
            dist.all_reduce(dist_tensor, op=dist.ReduceOp.AVG)
            global_function_value = dist_tensor[0]
        else:
            global_function_value = function_value
            
        # Keep everything on GPU to avoid CPU-GPU syncs!
        if not isinstance(group0['grad_l1_ema'], torch.Tensor):
            group0['grad_l1_ema'] = torch.tensor(group0['grad_l1_ema'], device=device)
        grad_l1_ema = group0['grad_l1_ema']
        
        grad_l1_ema.mul_(polyak_beta).add_(grad_l1, alpha=(1 - polyak_beta) * math.sqrt(math.pi / 2))
        
        if polyak_beta > 0:
            grad_l1_ema_corr = grad_l1_ema / (1 - polyak_beta ** (k + 1))
        else:
            grad_l1_ema_corr = grad_l1 * math.sqrt(math.pi / 2)
        
        polyak_lr = torch.where(
            grad_l1_ema_corr == 0,
            torch.ones_like(grad_l1_ema_corr),
            torch.clamp(global_function_value + ip_term, min=0.0) / grad_l1_ema_corr
        )
            
        for group in self.param_groups:
            eps = group['eps']
            lr = max(group['lr'], eps)
            decay = group['weight_decay']
            beta1, beta2 = group['betas']
            k = group['k']
            r = group['r']
            weight_lr_power = group['weight_lr_power']
            c_warmup = group['c_warmup']
            eta_scale = group['eta_scale']
            
            group_lr = lr * polyak_lr
            group['grad_l1_ema'] = grad_l1_ema
            group['scheduled_lr'] = group_lr
            
            if not isinstance(group['lr_max'], torch.Tensor):
                group['lr_max'] = torch.tensor(group['lr_max'], device=device)
            lr_max = group['lr_max'] = torch.maximum(group_lr, group['lr_max'])
            
            if k < c_warmup:
                ckp1 = 1.0
            else:
                weight = ((k + 1) ** r) * (lr_max ** weight_lr_power)
                if not isinstance(group['weight_sum'], torch.Tensor):
                    group['weight_sum'] = torch.tensor(group['weight_sum'], device=device)
                group['weight_sum'] = group['weight_sum'] + weight
                ckp1 = weight / group['weight_sum']
                
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]
                
                if 'z' not in state:
                    state['z'] = torch.clone(p.detach(), memory_format=torch.preserve_format)
                    state['x'] = torch.clone(p.detach(), memory_format=torch.preserve_format)
                    state['y'] = torch.clone(p.detach(), memory_format=torch.preserve_format)
                    
                    if p.ndim == 2 and group.get('use_aurora', True):
                        state['mom'] = torch.zeros_like(p.detach(), memory_format=torch.preserve_format)
                        state['vt'] = torch.zeros(p.size(0), 1, device=p.device, dtype=p.dtype)
                    else:
                        state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                        
                z, x, y = state['z'], state['x'], state['y']
                
                if p.ndim == 2 and group.get('use_aurora', True):
                    # SF-NorMuon Algorithm 1
                    mom = state['mom']
                    vt = state['vt']
                    
                    # Weight decay applied directly on Z at point y via fused add_
                    if decay > 0:
                        z.add_(y, alpha=-decay * group_lr * group_lr)
                    
                    # Explicit momentum before polar
                    mom.mul_(beta1).add_(grad, alpha=1 - beta1)
                    
                    # Spectral Step via fast Newton-Schulz
                    Pt = polar_fast(mom, eps=eps)
                    
                    # In-place update vt and get denom
                    P_hat_denom = _update_vt_and_get_denom(Pt, vt, beta2, eps)
                    
                    # In-place division: Pt becomes P_hat
                    Pt.div_(P_hat_denom)
                    Pt_norm = Pt.norm().clamp(min=1e-12)
                    
                    # Global scaling to match Adam RMS scale
                    m, n = p.shape
                    step_size = group_lr * (-eta_scale * math.sqrt(m * n))
                    step_size_scaled = step_size / Pt_norm
                    
                    # Update z with zero extra matrix allocations
                    z.add_(Pt, alpha=step_size_scaled)
                else:
                    # AdamC Polyak for 1D (Fully compiled & zero-allocation helper)
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    
                    bias_correction1 = 1 - beta1 ** (k + 1)
                    bias_correction2 = 1 - beta2 ** (k + 1)
                    
                    _update_1d_param(
                        z, y, grad, exp_avg, exp_avg_sq,
                        beta1, beta2, bias_correction1, bias_correction2,
                        group_lr, decay, eps
                    )
                    
                # Schedule-Free Extrapolation (Zero allocations)
                x.mul_(1.0 - ckp1).add_(z, alpha=ckp1)
                y.copy_(x).mul_(sf_beta1_k).add_(z, alpha=1.0 - sf_beta1_k)
                p.detach().copy_(y)
                
            group['k'] = k + 1
            
        return function_value
