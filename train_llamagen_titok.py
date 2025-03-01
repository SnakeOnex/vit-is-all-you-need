import torch, torch.nn as nn, torchvision, lpips
import argparse, tqdm, wandb, time
from einops import rearrange, repeat
from torch.amp import autocast, GradScaler
from dataclasses import dataclass

from utils import *
from train_vit import ViTConfig, ViT
from transformer import transformer_configs, Transformer
from datasets import get_imagenet_loaders, get_dmlab_image_loaders, get_minecraft_image_loaders

from LlamaGen.tokenizer.tokenizer_image.vq_model import VQ_models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

@dataclass
class TiTokConfig:
    vq_codebook_size: int
    vq_latent_tokens: int
    latent_tokens: int
    codebook_size: int
    latent_dim: int
    transformer: str
    def __post_init__(self):
        self.trans_config = transformer_configs[self.transformer](block_size=self.vq_latent_tokens+self.latent_tokens, dropout=0.0)
        self.n_embd = self.trans_config.n_embd

class TiTokEncoder(nn.Module):
    def __init__(self, titok_config: TiTokConfig):
        super(TiTokEncoder, self).__init__()
        self.latent_tokens = titok_config.latent_tokens
        self.transformer = Transformer(titok_config.trans_config)
        self.tok_emb = nn.Embedding(titok_config.vq_codebook_size, titok_config.n_embd)
        # self.pos_emb = nn.Embedding(titok_config.vq_latent_tokens, titok_config.n_embd)
        self.pos_emb = nn.Parameter(torch.randn(titok_config.vq_latent_tokens, titok_config.n_embd) * titok_config.n_embd ** -0.5)
        self.extra_emb = nn.Embedding(titok_config.latent_tokens, titok_config.n_embd)
        self.proj = nn.Linear(titok_config.n_embd, titok_config.latent_dim)
    def forward(self, x):
        input_emb = self.tok_emb(x) + self.pos_emb[torch.arange(x.shape[1], device=x.device)]

        extra_emb = repeat(self.extra_emb.weight, 'n d -> b n d', b=x.shape[0])
        input_emb = torch.cat([extra_emb, input_emb], dim=1)
        out_emb = self.transformer(input_emb)[:,:self.latent_tokens]
        latent_emb = self.proj(out_emb)
        return latent_emb

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
        # quantized = x

        return quantized, indices, quantize_loss

class TiTokDecoder(nn.Module):
    def __init__(self, titok_config: TiTokConfig):
        super(TiTokDecoder, self).__init__()
        self.config = titok_config
        self.transformer = Transformer(titok_config.trans_config)
        # self.pos_emb = nn.Embedding(titok_config.latent_tokens, titok_config.n_embd)
        self.pos_emb = nn.Parameter(torch.randn(titok_config.latent_tokens, titok_config.n_embd) * titok_config.n_embd ** -0.5)
        self.quant_proj = nn.Linear(titok_config.latent_dim, titok_config.n_embd)
        self.emb_proj = nn.Linear(titok_config.n_embd, titok_config.vq_codebook_size)
        self.mask_tokens = nn.Embedding(titok_config.vq_latent_tokens, titok_config.n_embd)
    def forward(self, z): # z = (B, latent_tokens, latent_dim)
        z_emb = self.quant_proj(z) + self.pos_emb[torch.arange(z.shape[1], device=z.device)]
        mask_emb = repeat(self.mask_tokens.weight, 'n d -> b n d', b=z.shape[0])
        emb = torch.cat([mask_emb, z_emb], dim=1)
        out_emb = self.transformer(emb)[:,:self.config.vq_latent_tokens]
        logits = self.emb_proj(out_emb)
        return logits

class TiTok(nn.Module):
    def __init__(self, titok_config: TiTokConfig):
        super(TiTok, self).__init__()
        self.config = titok_config
        self.enc = TiTokEncoder(titok_config)
        self.quant = Quantizer(titok_config)
        self.dec = TiTokDecoder(titok_config)

        self.apply(self._init_weights)
    def encode(self, z): return self.quant(self.enc(z))[1]
    def decode(self, z_quant): return self.dec(z_quant)
    def decode_indices(self, indices): return self.dec(self.quant.codebook(indices))
    def forward(self, x):
        latent_embs = self.enc(x)
        quantized, indices, quantize_loss = self.quant(latent_embs)
        codes_recon = self.dec(quantized)
        return codes_recon, indices, quantize_loss
    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None: module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vq_codebook_size', type=int, default=16384)
    parser.add_argument('--vq_latent_tokens', type=int, default=256)
    parser.add_argument('--latent_tokens', type=int, default=256)
    parser.add_argument('--codebook_size', type=int, default=16384)
    parser.add_argument('--latent_dim', type=int, default=12)
    parser.add_argument('--transformer', type=str, default='S')
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--micro_steps', type=int, default=1)
    parser.add_argument('--mixed', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--perceptual_weight', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--train_steps', type=int, default=1_000_000)
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--epochs', type=int, default=100000)
    args = parser.parse_args()
    args.min_lr = args.lr / 10.
    titok_config = TiTokConfig(args.vq_codebook_size, args.vq_latent_tokens, args.latent_tokens, args.codebook_size, args.latent_dim, args.transformer)

    if args.vq_latent_tokens == 256:
        vq_model = VQ_models['VQ-16']()
        vq_ckpt = "vq_ds16_c2i.pt"
    elif args.vq_latent_tokens == 1024:
        vq_model = VQ_models['VQ-8'](codebook_size=args.vq_codebook_size)
        if args.vq_codebook_size == 16384:
            vq_ckpt = "vq_ds8_c2i.pt"
        elif args.vq_codebook_size == 1024:
            vq_ckpt = "vq_ds8_c2i_1024.pt"
    elif args.vq_latent_tokens == 4096:
        vq_model = VQ_models['VQ-4'](codebook_size=args.vq_codebook_size)
        vq_ckpt = "vq_ds4_c2i.pt"

    vq_model.load_state_dict(torch.load(vq_ckpt, map_location="cpu")["model"])
    vq_model.eval()
    vq_model.to(device)

    if args.dataset == 'imagenet':
        project_name = 'titok-CE-imagenet'
        train_loader, _ = get_imagenet_loaders(256, args.bs // args.micro_steps)
    elif args.dataset == 'dmlab':
        assert args.image_size == 64
        project_name = 'titok-dmlab'
        train_loader, _ = get_dmlab_image_loaders(args.bs // args.micro_steps)
    elif args.dataset == 'minecraft':
        assert args.image_size == 128
        project_name = 'titok-minecraft'
        train_loader, _ = get_minecraft_image_loaders(args.bs // args.micro_steps)

    run_name=f"{args.vq_codebook_size}_{args.vq_latent_tokens}vq_{args.transformer}_{args.latent_tokens}_{args.codebook_size}ce_vq"

    wandb.init(project=project_name, name=run_name, config=titok_config.__dict__ | args.__dict__)

    titok = TiTok(titok_config).to(device)

    # DUMMY EXAMPLE
    # zq, _, info = vq_model.encode(torch.randn((8, 3, 256, 256)).to(device))
    # print(zq.shape)
    # vq_indices = rearrange(info[2], '(b z) -> b z', b=8)
    # print(f"{vq_indices.shape=}")
    #
    # recon, indices, q_loss = titok(vq_indices)
    # print(f"{recon.shape=}, {indices.shape=}, {q_loss=}")
    # recon_codes = recon.argmax(dim=-1)
    # recon_image = vq_model.decode_code(recon_codes, zq.shape)
    # print(recon_image.shape)

    optim = torch.optim.AdamW(titok.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = get_lr_scheduler(optim, args.warmup_steps, args.train_steps, args.min_lr)
    scaler = GradScaler(enabled=args.mixed)

    from perceptual_loss import PerceptualLoss
    lpips_loss_fn = PerceptualLoss().to(device).eval()
    loss_fn = nn.CrossEntropyLoss()

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
            with torch.no_grad():
                zq, _, info = vq_model.encode(images)
                vq_indices = rearrange(info[2], '(b z) -> b z', b=args.bs // args.micro_steps)
            load_time = time.time() - st
            optim.zero_grad()
            with autocast("cuda", enabled=args.mixed):
                codes_recon, indices, quantize_loss = titok(vq_indices)
                recon_loss = loss_fn(rearrange(codes_recon, 'b n c -> (b n) c'), rearrange(vq_indices, 'b n -> (b n)'))
                loss = recon_loss + quantize_loss
            scaler.scale(loss).backward()
            loss /= args.micro_steps
            micro_step += 1
            if micro_step != args.micro_steps: continue
            else: micro_step = 0

            scaler.step(optim)
            scaler.update()
            nn.utils.clip_grad_norm_(titok.parameters(), max_norm=1.0)
            lr_sched.step()
            step_time = time.time() - st - load_time
            codebook_usage[indices] = 1
            if step % 100 == 0: 
                codebook_usage_val = codebook_usage.sum().item() / titok_config.codebook_size
                # wandb.log({"train/epoch": epoch, "train/loss": loss.item(), "train/recon_loss": recon_loss.item(), "train/quant_loss": quantize_loss.item(), "train/perceptual_loss": perceptual_loss.item(), "train/l1_loss": l1_loss.item(), "train/codebook_usage": codebook_usage_val, "benchmark/load_time": load_time, "benchmark/step_time": step_time, "train/lr": optim.param_groups[0]['lr']})
                wandb.log({"train/epoch": epoch, "train/loss": loss.item(), "train/recon_loss": recon_loss.item(), "train/quant_loss": quantize_loss.item(), "train/codebook_usage": codebook_usage_val, "benchmark/load_time": load_time, "benchmark/step_time": step_time, "train/lr": optim.param_groups[0]['lr']})
                bar.set_description(f"e={epoch}: loss={loss.item():.3f} recon_loss={recon_loss.item():.3f} quant_loss={quantize_loss.item():.3f}")
                if recon_loss.item() < best_recon:
                    best_recon = recon_loss.item()
                    torch.save({"config": titok_config, "state_dict": titok.state_dict()}, f"titok_models/titok_{args.dataset}_{args.latent_tokens}_{args.codebook_size}.pt")
            if step % 5000 == 0: 
                code_preds = torch.argmax(codes_recon, dim=-1)
                code_preds = rearrange(code_preds, 'b n -> (b n)')
                with torch.no_grad():
                    print(f"{info[2].shape=} {zq.shape=} {code_preds.shape=}")
                    image_recon = vq_model.decode_code(info[2], zq.shape)[:4]
                    images_ce_recon = vq_model.decode_code(code_preds, zq.shape)[:4]
                images = [wandb.Image(img.permute(1, 2, 0).detach().cpu().numpy()) for img in images[:4]]
                recons = [wandb.Image(img.permute(1, 2, 0).detach().cpu().numpy()) for img in image_recon[:4]]
                ce_recons = [wandb.Image(img.permute(1, 2, 0).detach().cpu().numpy()) for img in images_ce_recon[:4]]
                codebook_usage *= 0
                wandb.log({"images": images, "reconstructions": recons, "ce_reconstructions": ce_recons})
            st = time.time()
            step += 1
        train_loss /= len(train_loader)

