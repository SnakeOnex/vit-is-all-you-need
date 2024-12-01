import torch, torch.nn as nn, torchvision
import argparse, tqdm, wandb, time
from einops import rearrange, repeat
from torch.amp import autocast, GradScaler
from dataclasses import dataclass
from transformer import Transformer, TransformerConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

@dataclass
class ViTConfig:
    image_size: int = 256
    in_channels: int = 3
    patch_size: int = 16
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768
    extra_tokens: int = 1
    dropout: float = 0.15

    def __post_init__(self):
        self.n_patches = (self.image_size // self.patch_size) ** 2
        self.patch_dim = 3 * self.patch_size ** 2

class ViT(nn.Module):
    def __init__(self, args: ViTConfig):
        super(ViT, self).__init__()
        self.config = args
        self.patch_proj = nn.Conv2d(in_channels=args.in_channels, out_channels=args.n_embd, kernel_size=args.patch_size, stride=args.patch_size)
        self.pos_emb = nn.Embedding(args.n_patches, args.n_embd)
        self.extra_emb = nn.Embedding(args.extra_tokens, args.n_embd)
        self.transformer = Transformer(TransformerConfig(args.n_layers, args.n_heads, args.n_embd, args.n_patches + args.extra_tokens, args.dropout))

    def forward(self, x):
        patch_emb = self.patch_proj(x)
        patch_emb = rearrange(patch_emb, 'b c h w -> b (h w) c')
        patch_emb = patch_emb + self.pos_emb(torch.arange(self.config.n_patches, device=patch_emb.device))

        extra_emb = repeat(self.extra_emb.weight, 'n d -> b n d', b=x.shape[0])
        emb = torch.cat([extra_emb, patch_emb], dim=1)
        return self.transformer(emb)

class ViTClassifier(nn.Module):
    def __init__(self, vit_config: ViTConfig, num_classes=1000):
        super(ViTClassifier, self).__init__()
        self.vit = ViT(vit_config)
        self.head = nn.Linear(vit_config.n_embd, num_classes)
    def forward(self, x):
        return self.head(self.vit(x)[:,0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/mnt/data/Public_datasets/imagenet/imagenet_pytorch')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--mixed', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--train_steps', type=int, default=200000)
    args = parser.parse_args()
    vit_config = ViTConfig(image_size=args.image_size)

    wandb.init(project="vit-classifier", config=vit_config.__dict__)

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.image_size),
        torchvision.transforms.RandomCrop(args.image_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.image_size),
        torchvision.transforms.CenterCrop(args.image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = torchvision.datasets.ImageNet(root=args.data_dir, split="train", transform=train_transform)
    valid_set = torchvision.datasets.ImageNet(root=args.data_dir, split="val", transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, prefetch_factor=2)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=2*args.bs, shuffle=False, num_workers=4, pin_memory=True)

    vit = ViTClassifier(vit_config).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(vit.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cos_lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.train_steps, eta_min=args.min_lr)
    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: min(1, s / args.warmup_steps))
    lr_sched = torch.optim.lr_scheduler.SequentialLR(optim, [warmup_sched, cos_lr_sched], [args.warmup_steps])
    scaler = GradScaler(enabled=args.mixed)

    # torch.compile(vit, mode="max-autotune")

    print(f"STATS: params={sum(p.numel() for p in vit.parameters())/1e6:.1f}M, trn_len={len(train_set)}, val_len={len(valid_set)}")
    print(f"PARAMS: {vit_config}")

    best_acc = 0.
    for epoch in range(100):
        bar = tqdm.tqdm(train_loader, disable=True)
        train_loss = 0.
        st = time.time()
        for i, (images, labels) in enumerate(bar):
            images, labels = images.to(device), labels.to(device)
            load_time = time.time() - st
            optim.zero_grad()
            with autocast("cuda", enabled=args.mixed):
                pred = vit(images)
                loss = loss_fn(pred, labels)
            train_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            lr_sched.step()
            step_time = time.time() - st - load_time
            if i % 100: wandb.log({"train/loss": loss.item(), "benchmark/load_time": load_time, "benchmark/step_time": step_time})
            bar.set_description(f"e={epoch} load_time={load_time:.3f}, step_time={step_time:.3f}")
            st = time.time()
        train_loss /= len(train_loader)

        with torch.no_grad():
            val_loss, acc = 0., 0.
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                vit.eval()
                pred = vit(images)
                vit.train()
                val_loss += loss_fn(pred, labels).item()
                acc += (pred.argmax(dim=-1) == labels).float().mean().item()
            val_loss /= len(valid_loader)
            acc /= len(valid_loader)
            print(f"epoch {epoch}: trn_loss={train_loss:.3f} val_loss={val_loss:.3f}, acc={acc:.3f}")
            wandb.log({"valid/loss": loss.item(), "valid/acc": acc})
            if acc > best_acc:
                best_acc = acc
                torch.save(vit.state_dict(), "vit.pth")

