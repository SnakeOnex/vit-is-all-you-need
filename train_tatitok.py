import torch, torch.nn as nn, torchvision, lpips
import argparse, tqdm, wandb, time
from einops import rearrange, repeat
from torch.amp import autocast, GradScaler
from dataclasses import dataclass

from utils import *
from train_vit import ViTConfig, ViT
from transformer import transformer_configs, Transformer
from datasets import get_imagenet_loaders, get_dmlab_image_loaders, get_minecraft_image_loaders

# from LlamaGen.tokenizer.tokenizer_image.vq_model import VQ_models
from blocks import TiTokEncoder, TiTokDecoder, VectorQuantizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

@dataclass
class TiTokConfig:
    image_size: int
    patch_size: int
    latent_tokens: int
    codebook_size: int
    latent_dim: int
    transformer: str
    use_l2_norm: bool = True

class TiTok(nn.Module):
    def __init__(self, config):

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config

        self.encoder = TiTokEncoder(config)
        self.decoder = TiTokDecoder(config)
        
        self.num_latent_tokens = config.latent_tokens
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        
        self.apply(self._init_weights)

        self.quantize = VectorQuantizer(
            codebook_size=config.codebook_size,
            token_size=config.latent_dim,
            commitment_cost=0.25,
            use_l2_norm=config.use_l2_norm,)
        
    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x):
        z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
        z_quantized, result_dict = self.quantize(z)

        return z_quantized, result_dict
    
    def decode(self, z_quantized):
        decoded = self.decoder(z_quantized)
        return decoded
    
    def decode_tokens(self, tokens):
        tokens = tokens.squeeze(1)
        batch, seq_len = tokens.shape # B x N
        z_quantized = self.quantize.get_codebook_entry(
            tokens.reshape(-1)).reshape(batch, 1, seq_len, -1)
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        decoded = self.decode(z_quantized)
        return decoded
    
    def forward(self, x):
        z_quantized, result_dict = self.encode(x)
        decoded = self.decode(z_quantized)
        return decoded, result_dict

def make_optim(model, args):
    # Exclude terms we may not want to apply weight decay.
    exclude = (lambda n, p: p.ndim < 2 or "ln" in n or "bias" in n or 'latent_tokens' in n 
               or 'mask_token' in n or 'embedding' in n or 'norm' in n or 'gamma' in n or 'embed' in n)
    include = lambda n, p: not exclude(n, p)
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": args.weight_decay},
        ],
        lr=args.lr,
        betas=(0.9, 0.999)
    )
    return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--latent_tokens', type=int, default=256)
    parser.add_argument('--codebook_size', type=int, default=16384)
    parser.add_argument('--latent_dim', type=int, default=12)
    parser.add_argument('--transformer', type=str, default='small')
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--micro_steps', type=int, default=1)
    parser.add_argument('--mixed', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--perceptual_weight', type=float, default=1.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--train_steps', type=int, default=1_000_000)
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--epochs', type=int, default=100000)
    args = parser.parse_args()
    args.min_lr = args.lr / 10.

    if args.dataset == 'imagenet':
        project_name = 'titok-single-imagenet'
        args.image_size = 256
        train_loader, _ = get_imagenet_loaders(args.image_size, args.bs // args.micro_steps)
    elif args.dataset == 'dmlab':
        assert args.image_size == 64
        project_name = 'titok-single-dmlab'
        args.image_size = 64
        train_loader, _ = get_dmlab_image_loaders(args.bs // args.micro_steps)
    elif args.dataset == 'minecraft':
        args.image_size = 128
        project_name = 'titok-single-minecraft'
        train_loader, _ = get_minecraft_image_loaders(args.bs // args.micro_steps)

    titok_config = TiTokConfig(args.image_size, args.patch_size, args.latent_tokens, args.codebook_size, args.latent_dim, args.transformer)

    run_name=f"{args.transformer}_{args.latent_tokens}_{args.codebook_size}"

    wandb.init(project=project_name, name=run_name, config=titok_config.__dict__ | args.__dict__)

    titok = TiTok(titok_config).to(device)
    wandb.watch(titok)

    # DUMMY EXAMPLE
    # zq, results_dict = titok(torch.randn((8, 3, 256, 256)).to(device))
    # print(results_dict.keys())
    # print(zq.shape)

    # optim = torch.optim.AdamW(titok.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    optim = make_optim(titok, args)
    lr_sched = get_lr_scheduler(optim, args.warmup_steps, args.train_steps, args.min_lr)
    scaler = GradScaler(enabled=args.mixed)

    from perceptual_loss import PerceptualLoss
    lpips_loss_fn = PerceptualLoss("lpips-convnext_s-1.0-0.1").to(device).eval()
    loss_fn = nn.MSELoss()

    print(f"STATS: enc_params={get_params_str(titok)}")

    best_recon = float('inf')
    for epoch in range(args.epochs):
        bar = tqdm.tqdm(train_loader)
        train_loss = 0.
        codebook_usage = torch.zeros([titok_config.codebook_size], device=device)
        st = time.time()

        step = 0
        micro_step = 0
        for images, _ in bar:
            images = images.to(device)
            load_time = time.time() - st
            with autocast("cuda", enabled=args.mixed):
                image_recon, results_dict = titok(images)
                quantize_loss = results_dict['quantizer_loss']
                l1_loss = (image_recon - images).pow(2).mean()
                perceptual_loss = args.perceptual_weight * lpips_loss_fn(image_recon, images).mean()
                recon_loss = l1_loss + perceptual_loss
                loss = recon_loss + quantize_loss
            scaler.scale(loss).backward()
            loss /= args.micro_steps
            micro_step += 1
            if micro_step != args.micro_steps: continue
            else: micro_step = 0

            nn.utils.clip_grad_norm_(titok.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.update()
            lr_sched.step()
            optim.zero_grad()
            step_time = time.time() - st - load_time
            codebook_usage[results_dict["min_encoding_indices"].flatten()] = 1
            if step % 100 == 0: 
                codebook_usage_val = codebook_usage.sum().item() / titok_config.codebook_size
                wandb.log({"train/epoch": epoch, "train/loss": loss.item(), "train/recon_loss": recon_loss.item(), "train/quant_loss": quantize_loss.item(), "train/perceptual_loss": perceptual_loss.item(), "train/l1_loss": l1_loss.item(), "train/codebook_usage": codebook_usage_val, "benchmark/load_time": load_time, "benchmark/step_time": step_time, "train/lr": optim.param_groups[0]['lr']})
                # wandb.log({"train/epoch": epoch, "train/loss": loss.item(), "train/recon_loss": recon_loss.item(), "train/quant_loss": quantize_loss.item(), "train/codebook_usage": codebook_usage_val, "benchmark/load_time": load_time, "benchmark/step_time": step_time, "train/lr": optim.param_groups[0]['lr']})
                bar.set_description(f"e={epoch}: loss={loss.item():.3f} recon_loss={recon_loss.item():.3f} quant_loss={quantize_loss.item():.3f}")
                if recon_loss.item() < best_recon:
                    best_recon = recon_loss.item()
                    torch.save({"config": titok_config, "state_dict": titok.state_dict()}, f"titok_models/titok_{args.dataset}_{args.latent_tokens}_{args.codebook_size}.pt")
            if step % 5000 == 0: 
                images = [wandb.Image(img.permute(1, 2, 0).detach().cpu().numpy()) for img in images[:4]]
                recons = [wandb.Image(img.permute(1, 2, 0).detach().cpu().numpy()) for img in image_recon[:4]]
                codebook_usage *= 0
                wandb.log({"images": images, "reconstructions": recons})
            st = time.time()
            step += 1
        train_loss /= len(train_loader)

