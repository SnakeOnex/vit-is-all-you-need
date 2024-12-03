import torch, torch.nn as nn, torch.nn.functional as F
import argparse, tqdm, wandb, time
from einops import rearrange
from torch.amp import autocast, GradScaler
from dataclasses import dataclass

from utils import *
from train_titok import TiTok, TiTokConfig
from transformer import Transformer, transformer_configs
from datasets import get_dmlab_video_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

@dataclass
class VideoGPTConfig:
    frame_size: int
    codebook_size: int
    transformer: str
    max_frames: int
    dropout: float
    def __post_init__(self):
        max_tokens = self.max_frames * self.frame_size
        self.trans_config = transformer_configs[self.transformer](block_size=max_tokens, dropout=self.dropout)
        self.n_embd = self.trans_config.n_embd

# two ways of looking at it: 
# as tokens: takes in N tokens and outputs (N+1)st token
# as frames: takes in N frames, (N x latent_size) and outputs (N+1)st (1 x latent_size) frame
# when implementing sliding window, we'll use the latter (there is never a need to teach the model to predict fraction of frames)
# video tokens: B x T x C x H x W -> B x T x N 
# when training the transformer autoregressive iter is:
#  - inp: B x (T*N-1+1) = SOS + video tokens of N frames, except the very last one
#  - out: B x (T*N) = video tokens of N frames
class VideoGPT(nn.Module):
    def __init__(self, config: VideoGPTConfig):
        super(VideoGPT, self).__init__()
        self.config = config
        self.tok_embed = nn.Embedding(config.codebook_size+1, config.n_embd)
        self.pos_embed = nn.Embedding(config.max_tokens, config.n_embd)
        self.transformer = Transformer(config.trans_config)
    def forward(self, x):
        B, T, N = x.shape
        sos = torch.zeros([B, 1], device=x.device, dtype=torch.long) + self.config.codebook_size
        y = rearrange(x, 'b t n -> b (t n)')
        x = torch.cat([sos, indices[:,:-1]], dim=-1)
        logits = self.transformer(x)
        return F.cross_entropy(logits, y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_size', type=int, default=256)
    parser.add_argument('--codebook_size', type=int, default=2048)
    parser.add_argument('--transformer', type=str, default='S')
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--mixed', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--train_steps', type=int, default=500000)
    parser.add_argument('--dataset', type=str, default='dmlab')
    args = parser.parse_args()
    args.min_lr = args.lr / 10.
    titok_config = TiTokConfig(args.image_size, args.patch_size, args.latent_tokens, args.codebook_size, args.latent_dim, args.transformer)

    project_name = f"videogpt-{args.dataset}"
    run_name=f"{args.frame_size}_{args.transformer}_{args.codebook_size}"
    if args.dataset == 'dmlab':
        assert args.image_size == 64
        train_loader, _ = get_dmlab_video_loaders(args.bs)
        titok = TiTok(titok_config).to(device)
        titok.load_state_dict(torch.load(f"titok_best.pth")) #TODO: load based on frame_size & codebook_size

    wandb.init(project=project_name, name=run_name, config=titok_config.__dict__ | args.__dict__)

    titok = TiTok(titok_config).to(device)

    optim = torch.optim.AdamW(titok.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = get_lr_scheduler(optim, args.warmup_steps, args.train_steps, args.min_lr)
    scaler = GradScaler(enabled=args.mixed)

    lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)

    print(f"STATS: enc_params={get_params_str(titok)}")

    best_recon = 0.
    for epoch in range(10000):
        bar = tqdm.tqdm(train_loader)
        train_loss = 0.
        codebook_usage = torch.zeros([titok_config.codebook_size], device=device)
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
            lr_sched.step()
            step_time = time.time() - st - load_time
            codebook_usage[indices] = 1
            if i % 100 == 0: 
                codebook_usage_val = codebook_usage.sum().item() / titok_config.codebook_size
                wandb.log({"train/loss": loss.item(), "train/recon_loss": recon_loss.item(), "train/quant_loss": quantize_loss.item(), "train/perceptual_loss": perceptual_loss.item(), "train/l1_loss": l1_loss.item(), "train/codebook_usage": codebook_usage_val, "benchmark/load_time": load_time, "benchmark/step_time": step_time, "train/lr": optim.param_groups[0]['lr']})
                bar.set_description(f"e={epoch}: loss={loss.item():.3f} recon_loss={recon_loss.item():.3f} quant_loss={quantize_loss.item():.3f}")
                if recon_loss.item() < best_recon:
                    best_recon = recon_loss.item()
                    torch.save(titok.state_dict(), f"titok_best_{run_name}.pth")
            if i % 5000 == 0: 
                images = [wandb.Image(img.permute(1, 2, 0).detach().cpu().numpy()) for img in images[:4]]
                recons = [wandb.Image(img.permute(1, 2, 0).detach().cpu().numpy()) for img in image_recon[:4]]
                codebook_usage *= 0
                wandb.log({"images": images, "reconstructions": recons})
            st = time.time()
        train_loss /= len(train_loader)

