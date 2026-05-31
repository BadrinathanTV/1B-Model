import math
import torch
import torch.distributed as dist
from .polar import polar_pe8


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
        
        function_value = None
        if closure is not None:
            function_value = closure()
        if function_value is None:
            function_value = 0.0 # Fallback
            
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
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                grad_l1_list.append(torch.linalg.vector_norm(grad, ord=1))
                state = self.state[p]
                if 'z' in state:
                    ip_term_list.append(sf_beta1_k * (grad.mul(state['z'] - state['x'])).sum())
                    
        grad_l1 = torch.stack(grad_l1_list).sum() if grad_l1_list else torch.tensor(0.0, device=self.param_groups[0]['params'][0].device)
        ip_term = torch.stack(ip_term_list).sum() if ip_term_list else torch.tensor(0.0, device=self.param_groups[0]['params'][0].device)
        
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(grad_l1, op=dist.ReduceOp.SUM)
            dist.all_reduce(ip_term, op=dist.ReduceOp.SUM)
            dist_tensor = torch.zeros(1, device=grad_l1.device)
            dist_tensor[0] = function_value
            dist.all_reduce(dist_tensor, op=dist.ReduceOp.AVG)
            global_function_value = dist_tensor[0].item()
        else:
            global_function_value = float(function_value)
            
        grad_l1 = grad_l1.item()
        ip_term = ip_term.item()
        
        grad_l1_ema = polyak_beta * grad_l1_ema + (1 - polyak_beta) * grad_l1 * math.sqrt(math.pi / 2)
        grad_l1_ema_corr = grad_l1_ema / (1 - polyak_beta ** (k + 1)) if polyak_beta > 0 else grad_l1 * math.sqrt(math.pi / 2)
        
        if grad_l1_ema_corr == 0:
            polyak_lr = 1.0
        else:
            polyak_lr = max(0.0, global_function_value + ip_term) / grad_l1_ema_corr
            
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
            lr_max = group['lr_max'] = max(group_lr, group['lr_max'])
            
            if k < c_warmup:
                ckp1 = 1.0
            else:
                weight = ((k + 1) ** r) * (lr_max ** weight_lr_power)
                weight_sum = group['weight_sum'] = group['weight_sum'] + weight
                ckp1 = weight / weight_sum
                
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
                    
                    # Weight decay applied directly on Z
                    if decay > 0:
                        z.sub_(z, alpha=group_lr * decay)
                    
                    # Explicit momentum before polar
                    mom.mul_(beta1).add_(grad, alpha=1 - beta1)
                    
                    # Spectral Step via Polar Express
                    Pt = polar_pe8(mom, eps)
                    
                    # Row-wise EMA normalization
                    Pt_sq = Pt.pow(2)
                    meancols = Pt_sq.mean(dim=1, keepdim=True)
                    vt.mul_(beta2).add_(meancols, alpha=1 - beta2)
                    
                    P_hat = Pt / (vt.sqrt() + eps)
                    
                    # Global scaling to match Adam RMS scale
                    m, n = p.shape
                    P_hat_norm = P_hat.float().norm().item()
                    eta_hat = eta_scale * group_lr * max(m, n) / max(1e-12, P_hat_norm)
                    
                    z.sub_(P_hat, alpha=eta_hat)
                else:
                    # AdamC Polyak for 1D
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    
                    # Fully decoupled AdamC weight decay on Z
                    if decay > 0:
                        z.sub_(z, alpha=group_lr * decay)
                    
                    bias_correction1 = 1 - beta1 ** (k + 1)
                    bias_correction2 = 1 - beta2 ** (k + 1)
                    
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_corr = exp_avg.div(bias_correction1)
                    
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)
                    
                    z.addcdiv_(exp_avg_corr, denom, value=-group_lr)
                    
                # Schedule-Free Extrapolation
                x.mul_(1 - ckp1).add_(z, alpha=ckp1)
                y.copy_(x.mul(sf_beta1_k).add_(z, alpha=1 - sf_beta1_k))
                p.detach().copy_(y)
                
            group['k'] = k + 1
            
        return function_value
