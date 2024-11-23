import torch, torch.nn as nn, torchvision
import argparse
from einops import rearrange, repeat
from dataclasses import dataclass

from transformer import Transformer, TransformerConfig

@dataclass
class ViTConfig:
    image_size: int = 256
    patch_size: int = 16
    num_classes: int = 1000
    n_layers: int = 12
    n_heads: int = 8
    n_embd: int = 1024
    extra_tokens: int = 1

    def __post_init__(self):
        self.n_patches = (self.image_size // self.patch_size) ** 2
        self.patch_dim = 3 * self.patch_size ** 2

class ViT(nn.Module):
    def __init__(self, args):
        super(ViT, self).__init__()
        self.config = args
        self.patch_proj = nn.Conv2d(in_channels=3, out_channels=args.n_embd, kernel_size=args.patch_size, stride=args.patch_size)
        self.pos_emb = nn.Embedding(args.n_patches, args.n_embd)
        self.extra_emb = nn.Embedding(args.extra_tokens, args.n_embd)
        self.transformer = Transformer(TransformerConfig(args.n_layers, args.n_heads, args.n_embd, args.n_patches + args.extra_tokens))

    def forward(self, x):
        patch_emb = self.patch_proj(x)
        patch_emb = rearrange(patch_emb, 'b c h w -> b (h w) c')
        patch_emb = patch_emb + self.pos_emb(torch.arange(self.config.n_patches, device=patch_emb.device))

        extra_emb = repeat(self.extra_emb.weight, 'n d -> b n d', b=x.shape[0])
        emb = torch.cat([extra_emb, patch_emb], dim=1)
        return self.transformer(emb)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/mnt/data/Public_datasets/imagenet/imagenet_pytorch')
    args = parser.parse_args()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(256),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # train_set, valid_set = [torchvision.datasets.ImageNet(root=args.data_dir, split=split, transform=transform) for split in ['train', 'val']]
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
    # valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=16, shuffle=False)

    vit = ViT(ViTConfig())

    images = torch.randn(2, 3, 256, 256)

    pred = vit(images)
    cls_out
    print(f"{pred.shape=}")
    exit(0)

    for images, labels in train_loader:
        print(images.shape)
        print(labels.shape)

        pred = vit(images)

        break
