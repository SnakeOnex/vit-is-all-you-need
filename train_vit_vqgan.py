import torch, torch.nn as nn, torchvision, lpips
import argparse, tqdm, wandb, time
from einops import rearrange
from torch.amp import autocast, GradScaler
from dataclasses import dataclass

from utils import *
from train_vit import ViTConfig, ViT
from datasets import get_imagenet_loaders, get_dmlab_image_loaders, get_minecraft_image_loaders
from vector_quantize_pytorch import FSQ

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

@dataclass
class ViTVQGANConfig:
    image_size: int
    patch_size: int
    codebook_size: int
    latent_dim: int
    transformer: str
    def __post_init__(self):
        self.patch_dim = self.image_size // self.patch_size
        self.n_patches = self.patch_dim**2
        self.latent_tokens = self.n_patches
        self.enc_vit_config = ViTConfig(self.image_size, 3, self.patch_size, self.transformer, 0, 0.0)
        self.n_embd = self.enc_vit_config.trans_config.n_embd
        self.dec_vit_config = ViTConfig(self.latent_tokens, self.n_embd, 1, self.transformer, 0, 0.0)
        self.dec_vit_config.n_patches = self.latent_tokens #HACK: is this line necessary?

class ViTVQGANEncoder(nn.Module):
    def __init__(self, config: ViTVQGANConfig):
        super(ViTVQGANEncoder, self).__init__()
        self.latent_tokens = config.latent_tokens
        self.vit = ViT(config.enc_vit_config)
        self.proj = nn.Linear(config.n_embd, config.latent_dim)
    def forward(self, x):
        out_embd = self.vit(x)
        latent_embd = self.proj(out_embd)
        return latent_embd

class Quantizer(nn.Module):
    def __init__(self, config: ViTVQGANConfig):
        super(Quantizer, self).__init__()
        self.codebook = nn.Embedding(config.codebook_size, config.latent_dim)
        self.codebook.weight.data.uniform_(-1.0 / config.codebook_size, 1.0 / config.codebook_size)
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

class ViTVQGANDecoder(nn.Module):
    def __init__(self, config: ViTVQGANConfig):
        super(ViTVQGANDecoder, self).__init__()
        self.config = config
        self.vit = ViT(config.dec_vit_config)
        self.quant_proj = nn.Linear(config.latent_dim, config.n_embd)
        self.embd_proj = nn.Conv2d(config.n_embd, 3*config.patch_size**2, kernel_size=1)
    def forward(self, z):
        z = self.quant_proj(z)
        z = rearrange(z, 'b h c -> b c h 1')
        out_embd = self.vit(z)
        out_embd = rearrange(out_embd, 'b (h w) c -> b c h w', h=self.config.patch_dim, w=self.config.patch_dim)
        image = self.embd_proj(out_embd)
        image = rearrange(image, 'b (p1 p2 c) h w -> b c (h p1) (w p2)', p1=self.config.patch_size, p2=self.config.patch_size)
        return image

class ViTVQGAN(nn.Module):
    def __init__(self, config: ViTVQGANConfig):
        super(ViTVQGAN, self).__init__()
        self.config = config
        self.encoder = ViTVQGANEncoder(config)
        self.quant = Quantizer(config)
        self.decoder = ViTVQGANDecoder(config)
    def encode(self, z): return self.quant(self.encoder(z))[1]
    def decode(self, z_quant): return self.decoder(z_quant)
    def decode_indices(self, indices): return self.decoder(self.quant.codebook(indices))
    def forward(self, x):
        latent_embs = self.encoder(x)
        quantized, indices, quantize_loss = self.quant(latent_embs)
        image_recon = self.decoder(quantized)
        return image_recon, indices, quantize_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--latent_tokens', type=int, default=256)
    parser.add_argument('--codebook_size', type=int, default=2048)
    parser.add_argument('--latent_dim', type=int, default=12)
    parser.add_argument('--transformer', type=str, default='B')
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--mixed', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--train_steps', type=int, default=500000)
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--epochs', type=int, default=100000)
    args = parser.parse_args()
    args.min_lr = args.lr / 10.
    config = ViTVQGANConfig(args.image_size, args.patch_size, args.codebook_size, args.latent_dim, args.transformer)

    if args.dataset == 'imagenet':
        project_name = 'vit-vqgan'
        train_loader, _ = get_imagenet_loaders(args.image_size, args.bs)
    # elif args.dataset == 'dmlab':
    #     assert args.image_size == 64
    #     project_name = 'titok-dmlab'
    #     train_loader, _ = get_dmlab_image_loaders(args.bs)
    # elif args.dataset == 'minecraft':
    #     assert args.image_size == 128
    #     project_name = 'titok-minecraft'
    #     train_loader, _ = get_minecraft_image_loaders(args.bs)

    run_name=f"{args.patch_size}px_{args.image_size}px_{args.transformer}_{args.latent_tokens}_{args.codebook_size}"

    wandb.init(project=project_name, name=run_name, config=config.__dict__ | args.__dict__)

    titok = ViTVQGAN(config).to(device)

    optim = torch.optim.AdamW(titok.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = get_lr_scheduler(optim, args.warmup_steps, args.train_steps, args.min_lr)
    scaler = GradScaler(enabled=args.mixed)

    from perceptual_loss import PerceptualLoss
    lpips_loss_fn = PerceptualLoss().to(device).eval()

    print(f"STATS: enc_params={get_params_str(titok)}")

    best_recon = float('inf')
    for epoch in range(args.epochs):
        bar = tqdm.tqdm(train_loader)
        train_loss = 0.
        codebook_usage = torch.zeros([config.codebook_size], device=device)
        st = time.time()
        for i, (images, _) in enumerate(bar):
            images = images.to(device)
            load_time = time.time() - st
            optim.zero_grad()
            with autocast("cuda", enabled=args.mixed):
                image_recon, indices, quantize_loss = titok(images)
                l1_loss = (image_recon - images).abs().mean()
                perceptual_loss = lpips_loss_fn(image_recon, images).mean()
                recon_loss = l1_loss + perceptual_loss
                loss = recon_loss + quantize_loss
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            nn.utils.clip_grad_norm_(titok.parameters(), max_norm=1.0)
            lr_sched.step()
            step_time = time.time() - st - load_time
            codebook_usage[indices] = 1
            if i % 500 == 0: 
                codebook_usage_val = codebook_usage.sum().item() / config.codebook_size
                wandb.log({"train/epoch": epoch, "train/loss": loss.item(), "train/recon_loss": recon_loss.item(), "train/quant_loss": quantize_loss.item(), "train/perceptual_loss": perceptual_loss.item(), "train/l1_loss": l1_loss.item(), "train/codebook_usage": codebook_usage_val, "benchmark/load_time": load_time, "benchmark/step_time": step_time, "train/lr": optim.param_groups[0]['lr']})
                bar.set_description(f"e={epoch}: loss={loss.item():.3f} recon_loss={recon_loss.item():.3f} quant_loss={quantize_loss.item():.3f}")
                if recon_loss.item() < best_recon:
                    best_recon = recon_loss.item()
                    torch.save({"config": config, "state_dict": titok.state_dict()}, f"titok_models/titok_{args.dataset}_{args.latent_tokens}_{args.codebook_size}.pt")
            if i % 5000 == 0: 
                images = [wandb.Image(img.permute(1, 2, 0).detach().cpu().numpy()) for img in images[:4]]
                recons = [wandb.Image(img.permute(1, 2, 0).detach().cpu().numpy()) for img in image_recon[:4]]
                codebook_usage *= 0
                wandb.log({"images": images, "reconstructions": recons})
            st = time.time()
        train_loss /= len(train_loader)

