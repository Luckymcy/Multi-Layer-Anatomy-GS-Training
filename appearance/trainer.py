"""Non-saturating GAN updates with strict discriminator/generator separation."""
import torch
from torch.nn import functional as F


def gan_step(lut, discriminator, lut_optimizer, d_optimizer, real, fake,
             tissue, smooth_weight, initial_rgb, anchor_weight=0., check_adversarial_gradient=False):
    lut_optimizer.zero_grad(set_to_none=True)
    d_optimizer.zero_grad(set_to_none=True)
    discriminator.train()
    discriminator.requires_grad_(True)
    real_logits = discriminator(real, tissue)
    fake_logits = discriminator(fake.detach(), tissue)
    d_loss = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
    if not torch.isfinite(d_loss):
        raise RuntimeError('Nonfinite discriminator loss')
    d_loss.backward()
    if lut.logits.grad is not None:
        raise RuntimeError('Discriminator update leaked into LUT')
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in discriminator.parameters()):
        raise RuntimeError('Nonfinite discriminator gradient')
    d_optimizer.step()
    d_optimizer.zero_grad(set_to_none=True)

    discriminator.requires_grad_(False)
    discriminator.eval()
    # No detach/no_grad here: D input -> render -> LUT must remain differentiable.
    adversarial = F.softplus(-discriminator(fake, tissue)).mean()
    adv_gradient_norm = None
    if check_adversarial_gradient:
        adv_grad = torch.autograd.grad(adversarial, lut.logits, retain_graph=True)[0]
        if not torch.isfinite(adv_grad).all() or not adv_grad.abs().max() > 0:
            raise RuntimeError('Missing/nonfinite adversarial -> LUT gradient')
        adv_gradient_norm = float(adv_grad.norm())
    smooth = lut.smoothness()
    anchor = (lut.rgb-initial_rgb).square().mean()
    g_loss = adversarial + smooth_weight*smooth + anchor_weight*anchor
    if not torch.isfinite(g_loss):
        raise RuntimeError('Nonfinite LUT loss')
    g_loss.backward()
    if lut.logits.grad is None or not torch.isfinite(lut.logits.grad).all():
        raise RuntimeError('Missing/nonfinite LUT gradient')
    if any(p.grad is not None for p in discriminator.parameters()):
        raise RuntimeError('Frozen discriminator received gradients during LUT update')
    grad_norm = float(lut.logits.grad.norm())
    lut_optimizer.step()
    return {'d_loss': float(d_loss), 'g_loss': float(g_loss), 'g_adversarial': float(adversarial),
            'lut_smoothness': float(smooth), 'lut_anchor': float(anchor),
            'lut_gradient_norm': grad_norm, 'adversarial_gradient_norm': adv_gradient_norm,
            'd_real_logit_mean': float(real_logits.detach().mean()),
            'd_fake_logit_mean': float(fake_logits.detach().mean())}
