import torch, torchvision

def get_imagenet_loaders(image_size, bs, data_dir='/mnt/data/Public_datasets/imagenet/imagenet_pytorch'):
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.RandomCrop(image_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.CenterCrop(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = torchvision.datasets.ImageNet(root=data_dir, split="train", transform=train_transform)
    valid_set = torchvision.datasets.ImageNet(root=data_dir, split="val", transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, prefetch_factor=2)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=2*bs, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, valid_loader
