import numpy as np, torch, torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from einops import rearrange

def get_imagenet_loaders(image_size, bs, data_dir='/mnt/data/Public_datasets/imagenet/imagenet_pytorch'):
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = torchvision.datasets.ImageNet(root=data_dir, split="train", transform=train_transform)
    valid_set = torchvision.datasets.ImageNet(root=data_dir, split="val", transform=valid_transform)

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, prefetch_factor=2)
    valid_loader = DataLoader(valid_set, batch_size=2*bs, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, valid_loader


class DmlabDataset(Dataset):
    def __init__(self, dataset_path):
        self.video_paths = []
        for folder_path in Path(dataset_path).iterdir():
            for video_path in folder_path.iterdir():
                self.video_paths.append(video_path)
    def __len__(self): return len(self.video_paths)
    def __getitem__(self, idx):
        data = np.load(self.video_paths[idx])
        video, action = data['video'], data['actions']
        # video = (torch.from_numpy(video).float() / 255) * 2 - 1
        # video = video.permute(0, 3, 1, 2)
        action = torch.from_numpy(action)
        return video, action

class MinecraftDataset(Dataset):
    def __init__(self, dataset_path):
        self.video_paths = []
        for folder_path in Path(dataset_path).iterdir():
            for video_path in folder_path.iterdir():
                if video_path.suffix == '.mp4':
                    self.video_paths.append(video_path)
    def __len__(self): return len(self.video_paths)
    def __getitem__(self, idx):
        video, audio, info = torchvision.io.read_video(self.video_paths[idx])
        # video, _ = data['video'], data['actions']
        video = (video.float() / 255) * 2 - 1
        video = video.permute(0, 3, 1, 2)
        # action = torch.from_numpy(action)
        return video, None

class ImagesFromVideoDataset(Dataset):
    def __init__(self, video_dataset, frames_per_video=8):
        self.video_dataset = video_dataset
        self.frames_per_video = frames_per_video
    def __len__(self): return len(self.video_dataset)*1000
    def __getitem__(self, idx):
        video, _ = self.video_dataset[idx % len(self.video_dataset)]
        frames = torch.stack([video[torch.randint(0, video.shape[0], (self.frames_per_video,))]]).squeeze(0)
        return frames, None

def collate_fn(batch):
    x, y = zip(*batch)
    x = torch.stack(x)
    x = rearrange(x, "b f c h w -> (b f) c h w")
    return x, None

# very simple dataloader that samples random frames from random videos
def video_dataloader(dataset, batch_size, videos_per_batch=4):
    while True:
        # 1. fetch videos_per_batch videos
        videos = torch.stack([dataset[i][0] for i in np.random.choice(len(dataset), videos_per_batch)])
        # 2. sample batch_size / videos_per_batch frames from each video
        frames_per_video = batch_size // videos_per_batch
        frames = torch.stack([video[torch.randint(0, video.shape[0], (frames_per_video,))] for video in videos])
        frames = rearrange(frames, "b f c h w -> (b f) c h w")
        yield frames, None

def get_dmlab_image_loaders(batch_size, dataset_path='../teco/dmlab/train/'):
    dataset = DmlabDataset(dataset_path)
    dataset = ImagesFromVideoDataset(dataset)
    # loader = video_dataloader(dataset, batch_size)
    loader = DataLoader(dataset, batch_size=batch_size//8, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate_fn, prefetch_factor=2)
    return loader, None

def get_minecraft_image_loaders(batch_size, dataset_path='../teco/minecraft/train/'):
    dataset = MinecraftDataset(dataset_path)
    dataset = ImagesFromVideoDataset(dataset)
    loader = DataLoader(dataset, batch_size=batch_size//8, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate_fn, prefetch_factor=2)
    return loader, None

def get_dmlab_video_loaders(batch_size, dataset_path='../teco/dmlab/train/'):
    dataset = DmlabDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    return loader, None
