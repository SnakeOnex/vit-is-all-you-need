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
        self.max_tokens = self.max_frames * self.frame_size
        self.trans_config = transformer_configs[self.transformer](block_size=self.max_tokens, dropout=self.dropout, causal=True)
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
        x = torch.cat([sos, y[:,:-1]], dim=-1)
        x = self.tok_embed(x) + self.pos_embed(torch.arange(T*N, device=x.device))
        logits = self.transformer(x)
        loss = F.cross_entropy(rearrange(logits, 'b s d -> (b s) d'), rearrange(y, 'b s -> (b s)'))
        return logits, loss
    def generate(self, tokens, n=1):
        for _ in range(n):
            sos = torch.zeros([tokens.shape[0], 1], device=tokens.device, dtype=torch.long) + self.config.codebook_size
            x = torch.cat([sos, tokens], dim=-1)
            x = self.tok_embed(x) + self.pos_embed(torch.arange(x.shape[1], device=x.device))
            logits = self.transformer(x)
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=-1)
        return tokens

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_size', type=int, default=16)
    parser.add_argument('--codebook_size', type=int, default=512)
    parser.add_argument('--transformer', type=str, default='S')
    parser.add_argument('--max_frames', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--mixed', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--train_steps', type=int, default=500000)
    parser.add_argument('--dataset', type=str, default='dmlab')
    parser.add_argument('--epochs', type=int, default=100000)
    args = parser.parse_args()
    args.min_lr = args.lr / 10.
    videogpt_config = VideoGPTConfig(args.frame_size, args.codebook_size, args.transformer, args.max_frames, args.dropout)

    project_name = f"videogpt-{args.dataset}"
    run_name=f"{args.frame_size}_{args.transformer}_{args.codebook_size}"
    if args.dataset == 'dmlab':
        train_loader, _ = get_dmlab_video_loaders(args.bs)
        titok_params = torch.load(f"titok_models/titok_{args.dataset}_{args.frame_size}_{args.codebook_size}.pt")
        titok = TiTok(titok_params["config"]).to(device).eval()
        titok.load_state_dict(titok_params["state_dict"])
        print(f"{titok_params['config']}")

    wandb.init(project=project_name, name=run_name, config=args.__dict__)

    video_gpt = VideoGPT(videogpt_config).to(device)

    optim = torch.optim.AdamW(video_gpt.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = get_lr_scheduler(optim, args.warmup_steps, args.train_steps, args.min_lr)
    scaler = GradScaler(enabled=args.mixed)

    print(f"STATS: titok_params={get_params_str(titok)}, video_gpt_params={get_params_str(video_gpt)}")

    best_recon = 0.
    for epoch in range(args.epochs):
        bar = tqdm.tqdm(train_loader)
        st = time.time()
        for i, (videos, _) in enumerate(bar):
            videos = videos.to(device)
            videos = videos[:, :args.max_frames]
            B, T, C, H, W = videos.shape

            # 1. encode video into tokens
            videos = rearrange(videos, 'b t c h w -> (b t) c h w')
            with torch.no_grad(): tokens = titok.encode(videos)
            tokens = rearrange(tokens, '(b t) n -> b t n', b=B)
            load_time = time.time() - st
            tokens = tokens[:, :1]
            print(tokens.shape)
            with torch.no_grad(): 
                out_tokens = video_gpt.generate(rearrange(tokens, 'b t n -> b (t n)'), n=5)
                print(out_tokens.shape)
            exit(0)
            optim.zero_grad()
            with autocast("cuda", enabled=args.mixed):
                _, loss = video_gpt(tokens)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            lr_sched.step()
            step_time = time.time() - st - load_time
            if i % 100 == 0: 
                wandb.log({"train/loss": loss.item(), "benchmark/load_time": load_time, "benchmark/step_time": step_time, "train/lr": optim.param_groups[0]['lr'], "train/epoch": epoch})
                bar.set_description(f"e={epoch}: loss={loss.item():.3f}")
            # if i % 5000 == 0: 
            #     images = [wandb.Image(img.permute(1, 2, 0).detach().cpu().numpy()) for img in images[:4]]
            #     recons = [wandb.Image(img.permute(1, 2, 0).detach().cpu().numpy()) for img in image_recon[:4]]
            #     codebook_usage *= 0
            #     wandb.log({"images": images, "reconstructions": recons})
            st = time.time()

