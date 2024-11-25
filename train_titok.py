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
    image_sz = 128
    latent_tokens: int = 32
    codebook_size: int = 512
    latent_dim: int = 256

# image(3, 128, 128) -> TiTokEncoder(128/16^2, n_embed) -> latent(latent_tokens, latent_dim) -> Quantizer(latent_tokens, latent_dim) -> TiTokDecoder(128/16^2, n_embed) -> image(3, 128, 128)

class TiTokEncoder(nn.Module):
    def __init__(self, titok_config: TiTokConfig):
        super(TiTokEncoder, self).__init__()
        self.latent_tokens = titok_config.latent_tokens
        self.vit = ViT(ViTConfig(image_size=titok_config.image_sz, extra_tokens=titok_config.latent_tokens))
    def forward(self, x):
        out_embd = self.vit(x)
        return out_embd

class Quantizer(nn.Module):
    def __init__(self, titok_config: TiTokConfig):
        super(Quantizer, self).__init__()
        self.codebook = nn.Embedding(titok_config.codebook_size, titok_config.latent_dim)
    def forward(self, x):
        pass

class TiTokDecoder(nn.Module):
    def __init__(self, titok_config: TiTokConfig):
        super(TiTokDecoder, self).__init__()
    def forward(self, x):
        pass

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

    wandb.init(project="vit-classifier", config=vit_config.__dict__)

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.image_size),
        torchvision.transforms.RandomCrop(args.image_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.image_size),
        torchvision.transforms.CenterCrop(args.image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_set = torchvision.datasets.ImageNet(root=args.data_dir, split="train", transform=train_transform)
    valid_set = torchvision.datasets.ImageNet(root=args.data_dir, split="val", transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, prefetch_factor=2)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=2*args.bs, shuffle=False, num_workers=4, pin_memory=True)

    vit = ViTClassifier(vit_config).to(device)
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

