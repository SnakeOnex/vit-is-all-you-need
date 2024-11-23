import torch, torchvision
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/mnt/data/Public_datasets/imagenet/imagenet_pytorch')
    args = parser.parse_args()

    train_set, valid_set = [torchvision.datasets.ImageNet(root=args.data_dir, split=split) for split in ['train', 'val']]
    train_loader, valid_loader = [torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True) for dataset in [train_set, valid_set]]

    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    #
    # for images, labels in dataloader:
    #     print(images.shape)
    #     print(labels)
    #     break
