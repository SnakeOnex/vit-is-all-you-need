import torch, torch.nn as nn, torchvision
import argparse, tqdm, wandb
from einops import rearrange, repeat
from dataclasses import dataclass

from transformer import Transformer, TransformerConfig
from train_vit import ViTConfig, ViT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ViTClassifier(nn.Module):
    def __init__(self, vit_config: ViTConfig):
        super(ViTClassifier, self).__init__()
        self.vit = ViT(vit_config)
        self.head = nn.Linear(vit_config.n_embd, vit_config.num_classes)
    def forward(self, x):
        return self.head(self.vit(x)[:,0])

@dataclass
class TiTokConfig:
    image_sz: int = 128
    latent_tokens: int = 32
    codebook_size: int = 512
    latent_dim: int = 256

# image(3, 128, 128) -> TiTokEncoder(128/16^2, n_embed) -> latent(latent_tokens, latent_dim) -> Quantizer(latent_tokens, latent_dim) -> TiTokDecoder(128/16^2, n_embed) -> image(3, 128, 128)

class TiTokEncoder(nn.Module):
    def __init__(self, titok_config: TiTokConfig):
        super(TiTokEncoder, self).__init__()
        self.latent_tokens = titok_config.latent_tokens
        self.vit = ViT(ViTConfig(image_size=titok_config.image_sz, extra_tokens=titok_config.latent_tokens))
        self.proj = nn.Linear(self.vit.config.n_embd, titok_config.latent_dim)
    def forward(self, x):
        out_embd = self.vit(x)
        latent_embd = self.proj(out_embd)
        return latent_embd

class Quantizer(nn.Module):
    def __init__(self, titok_config: TiTokConfig):
        super(Quantizer, self).__init__()
        self.codebook = nn.Embedding(titok_config.codebook_size, titok_config.latent_dim)
    def forward(self, x):
        nearest_neighbors = torch.cdist(x, self.codebook.weight)
        indices = nearest_neighbors.argmin(dim=-1)
        quantized = x + (self.codebook(indices) - x).detach()
        return quantized, indices

class TiTokDecoder(nn.Module):
    def __init__(self, titok_config: TiTokConfig):
        super(TiTokDecoder, self).__init__()
        self.config = titok_config
        vit_config = ViTConfig(image_size=titok_config.image_sz, extra_tokens=(titok_config.image_sz//16)**2)
        self.quant_proj = nn.Linear(titok_config.latent_dim, vit_config.n_embd)
        vit_config = ViTConfig(image_size=titok_config.image_sz, extra_tokens=titok_config.latent_tokens)
        vit_config.patch_size = 1
        vit_config.in_channels = vit_config.n_embd
        vit_config.n_patches = 32
        self.vit = ViT(vit_config)
        self.embd_proj = nn.Conv2d(vit_config.n_embd, 16*16*3, kernel_size=1)
    def forward(self, x):
        print(f"{x.shape=}")
        x = self.quant_proj(x)
        print(f"{x.shape=}")
        x = rearrange(x, 'b h c -> b c h 1')
        print(f"{x.shape=}")
        out_embd = self.vit(x)
        print(f"{out_embd.shape=}")
        out_embd = rearrange(out_embd, 'b (h w) c -> b c h w', h=self.config.image_sz//16, w=self.config.image_sz//16)
        print(f"{out_embd.shape=}")
        image = self.embd_proj(out_embd)
        print(f"{image.shape=}")
        image = rearrange(image, 'b (p1 p2 c) h w -> b c (h p1) (w p2)', p1=16, p2=16)
        print(f"{image.shape=}")
        exit(0)
        out_embd = self.vit(x)
        exit(0)
        print(f"{out_embd.shape=}")
        exit(0)
        # out_embd = rearrange(out_embd, 'b (h w) c -> b c h w', h=self.config.image_sz//16, w=self.config.image_sz//16)
        print(f"{out_embd.shape=}")
        exit(0)

# class ViT(nn.Module):
#     def __init__(self, args: ViTConfig):
#         super(ViT, self).__init__()
#         self.config = args
#         # self.patch_proj = nn.Linear(args.wi
#         self.pos_emb = nn.Embedding(args.n_patches, args.n_embd)
#         self.extra_emb = nn.Embedding(args.extra_tokens, args.n_embd)
#         self.transformer = Transformer(TransformerConfig(args.n_layers, args.n_heads, args.n_embd, args.n_patches + args.extra_tokens, args.dropout))
#
#     def forward(self, x):
#         patch_emb = self.patch_proj(x)
#         patch_emb = rearrange(patch_emb, 'b c h w -> b (h w) c')
#         patch_emb = patch_emb + self.pos_emb(torch.arange(self.config.n_patches, device=patch_emb.device))
#
#         extra_emb = repeat(self.extra_emb.weight, 'n d -> b n d', b=x.shape[0])
#         emb = torch.cat([extra_emb, patch_emb], dim=1)
#         return self.transformer(emb)


class TiTok(nn.Module):
    def __init__(self, titok_config: TiTokConfig, vit_config: ViTConfig):
        super(TiTok, self).__init__()
        self.vit_encoder = ViT(vit_config, extra_tokens=titok_config.latent_tokens)
    def forward(self, x):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/mnt/data/Public_datasets/imagenet/imagenet_pytorch')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--bs', type=int, default=128)
    args = parser.parse_args()
    vit_config = ViTConfig(image_size=args.image_size)
    titok_config = TiTokConfig(image_sz=args.image_size)

    wandb.init(project="vit-classifier", config=vit_config.__dict__)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.image_size),
        torchvision.transforms.CenterCrop(args.image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # train_set = torchvision.datasets.ImageNet(root=args.data_dir, split="train", transform=transform)
    # valid_set = torchvision.datasets.ImageNet(root=args.data_dir, split="val", transform=transform)
    #
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, prefetch_factor=2)
    # valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=2*args.bs, shuffle=False, num_workers=4, pin_memory=True)

    titok_enc = TiTokEncoder(titok_config).to(device)
    quantizer = Quantizer(titok_config).to(device)
    titok_dec = TiTokDecoder(titok_config).to(device)

    images = torch.randn(2, 3, args.image_size, args.image_size).to(device)
    latent_embs = titok_enc(images)[:,:titok_config.latent_tokens]
    print(f"{latent_embs.shape=}")
    quantized, indices = quantizer(latent_embs)
    print(f"{quantized.shape=}, {indices.shape=}")
    image_recon = titok_dec(quantized)
    print(f"{image_recon.shape=}")
    exit(0)

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(vit.parameters(), lr=1e-4, weight_decay=1e-3)


    print(f"STATS: params={sum(p.numel() for p in vit.parameters())/1e6:.1f}M, trn_len={len(train_set)}, val_len={len(valid_set)}")
    print(f"PARAMS: {vit_config}")

    best_acc = 0.
    for epoch in range(100):
        bar = tqdm.tqdm(train_loader)
        train_loss = 0.
        for i, (images, labels) in enumerate(bar):
            images, labels = images.to(device), labels.to(device)
            optim.zero_grad()
            pred = vit(images)
            loss = loss_fn(pred, labels)
            train_loss += loss.item()
            loss.backward()
            optim.step()
            bar.set_description(f"e={epoch} loss={loss.item():.3f}")
            if i % 10 == 0: wandb.log({"train/loss": loss.item()})
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

