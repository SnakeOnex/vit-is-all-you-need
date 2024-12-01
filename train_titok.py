import torch, torch.nn as nn, torchvision, lpips
import argparse, tqdm, wandb, time
from einops import rearrange
from torch.amp import autocast, GradScaler
from dataclasses import dataclass

from utils import *
from train_vit import ViTConfig, ViT
from datasets import get_imagenet_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

@dataclass
class TiTokConfig:
    image_sz: int = 128
    patch_size: int = 16
    latent_tokens: int = 256
    codebook_size: int = 2048
    latent_dim: int = 12

class TiTokEncoder(nn.Module):
    def __init__(self, titok_config: TiTokConfig):
        super(TiTokEncoder, self).__init__()
        self.latent_tokens = titok_config.latent_tokens
        self.vit = ViT(ViTConfig(image_size=titok_config.image_sz, patch_size=titok_config.patch_size, extra_tokens=titok_config.latent_tokens))
        self.proj = nn.Linear(self.vit.config.n_embd, titok_config.latent_dim)
    def forward(self, x):
        out_embd = self.vit(x)
        latent_embd = self.proj(out_embd)
        return latent_embd

class Quantizer(nn.Module):
    def __init__(self, titok_config: TiTokConfig):
        super(Quantizer, self).__init__()
        self.codebook = nn.Embedding(titok_config.codebook_size, titok_config.latent_dim)
        self.codebook.weight.data.uniform_(-1.0 / titok_config.codebook_size, 1.0 / titok_config.codebook_size)
    def forward(self, x):
        x = torch.nn.functional.normalize(x, dim=-1)
        embedding = torch.nn.functional.normalize(self.codebook.weight, dim=-1)
        indices = torch.cdist(x, embedding).argmin(dim=-1)
        quantized = self.codebook(indices)
        codebook_loss = (quantized - x.detach()).pow(2).mean()
        commitment_loss = 0.25 * (quantized.detach() - x).pow(2).mean()
        quantize_loss = codebook_loss + commitment_loss
        quantized = x + (quantized - x).detach() # copy gradients
        return quantized, indices, quantize_loss

class TiTokDecoder(nn.Module):
    def __init__(self, titok_config: TiTokConfig):
        super(TiTokDecoder, self).__init__()
        self.config = titok_config
        vit_config = ViTConfig(image_size=titok_config.image_sz, extra_tokens=(titok_config.image_sz//16)**2)
        vit_config.patch_size = 1
        vit_config.in_channels = vit_config.n_embd
        vit_config.n_patches = titok_config.latent_tokens
        self.quant_proj = nn.Linear(titok_config.latent_dim, vit_config.n_embd)
        self.vit = ViT(vit_config)
        self.embd_proj = nn.Conv2d(vit_config.n_embd, 3*titok_config.patch_size**2, kernel_size=1)
        self.conv_out = nn.Conv2d(3, 3, 3, padding=1, bias=True) # one more conv as per the original paper
    def forward(self, x):
        x = self.quant_proj(x)
        x = rearrange(x, 'b h c -> b c h 1')
        out_embd = self.vit(x)[:,:(self.config.image_sz//self.config.patch_size)**2]
        out_embd = rearrange(out_embd, 'b (h w) c -> b c h w', h=self.config.image_sz//self.config.patch_size, w=self.config.image_sz//self.config.patch_size)
        image = self.embd_proj(out_embd)
        image = rearrange(image, 'b (p1 p2 c) h w -> b c (h p1) (w p2)', p1=self.config.patch_size, p2=self.config.patch_size)
        image = self.conv_out(image)
        return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--mixed', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--train_steps', type=int, default=200000)
    args = parser.parse_args()
    vit_config = ViTConfig(image_size=args.image_size, dropout=0.0)
    titok_config = TiTokConfig(image_sz=args.image_size)

    wandb.init(project="titok", config=titok_config.__dict__ | vit_config.__dict__ | args.__dict__)

    titok_enc = TiTokEncoder(titok_config).to(device)
    quantizer = Quantizer(titok_config).to(device)
    titok_dec = TiTokDecoder(titok_config).to(device)

    train_loader, valid_loader = get_imagenet_loaders(args.image_size, args.bs)

    params = list(titok_enc.parameters()) + list(quantizer.parameters()) + list(titok_dec.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    cos_lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.train_steps, eta_min=args.min_lr)
    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: min(1, s / args.warmup_steps))
    lr_sched = torch.optim.lr_scheduler.SequentialLR(optim, [warmup_sched, cos_lr_sched], [args.warmup_steps])
    scaler = GradScaler(enabled=args.mixed)

    lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)

    print(f"STATS: enc_params={get_params_str(titok_enc)}, \
            dec_params={get_params_str(titok_dec)} \
            trn_len={len(train_loader.dataset)}, val_len={len(valid_loader.dataset)}")

    best_acc = 0.
    for epoch in range(10000):
        bar = tqdm.tqdm(train_loader)
        train_loss = 0.
        codebook_usage = torch.zeros([titok_config.codebook_size], device=device)
        st = time.time()
        for i, (images, labels) in enumerate(bar):
            images, labels = images.to(device), labels.to(device)
            load_time = time.time() - st
            optim.zero_grad()
            with autocast("cuda", enabled=args.mixed):
                latent_embs = titok_enc(images)[:,:titok_config.latent_tokens]
                quantized, indices, quantize_loss = quantizer(latent_embs)
                image_recon = titok_dec(quantized)

                l1_loss = (image_recon - images).abs().mean()
                perceptual_loss = lpips_loss_fn(image_recon, images).mean()
                recon_loss = l1_loss + perceptual_loss
                loss = recon_loss + quantize_loss
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            lr_sched.step()
            step_time = time.time() - st - load_time
            codebook_usage[indices] = 1
            if i % 100 == 0: 
                codebook_usage_val = codebook_usage.sum().item() / titok_config.codebook_size
                wandb.log({"train/loss": loss.item(), "train/recon_loss": recon_loss.item(), "train/quant_loss": quantize_loss.item(), "train/perceptual_loss": perceptual_loss.item(), "train/l1_loss": l1_loss.item(), "train/codebook_usage": codebook_usage_val, "benchmark/load_time": load_time, "benchmark/step_time": step_time, "train/lr": optim.param_groups[0]['lr']})
                bar.set_description(f"e={epoch}: loss={loss.item():.3f} recon_loss={recon_loss.item():.3f} quant_loss={quantize_loss.item():.3f}")
            if i % 5000 == 0: 
                images = [wandb.Image(img.permute(1, 2, 0).detach().cpu().numpy()) for img in images[:4]]
                recons = [wandb.Image(img.permute(1, 2, 0).detach().cpu().numpy()) for img in image_recon[:4]]
                codebook_usage *= 0
                wandb.log({"images": images, "reconstructions": recons})
            st = time.time()
        train_loss /= len(train_loader)

