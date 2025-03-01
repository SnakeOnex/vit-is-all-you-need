# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/webdataset/webdataset-imagenet/blob/main/convert-imagenet.py

import argparse
import os
import sys
import time
import torch
import tqdm
import numpy as np
from PIL import Image

import webdataset as wds
# from datasets import load_dataset
from datasets import DmlabDataset, MinecraftDataset, UCF_101_Dataset


def convert_imagenet_to_wds(output_dir, max_train_samples_per_shard, max_val_samples_per_shard):
    assert not os.path.exists(os.path.join(output_dir, "imagenet-train-000000.tar"))
    assert not os.path.exists(os.path.join(output_dir, "imagenet-val-000000.tar"))

    opat = os.path.join(output_dir, "imagenet-train-%06d.tar")
    output = wds.ShardWriter(opat, maxcount=max_train_samples_per_shard)
    dataset = load_dataset("imagenet-1k", streaming=True, split="train", use_auth_token=True)
    now = time.time()
    for i, example in enumerate(dataset):
        if i % max_train_samples_per_shard == 0:
            print(i, file=sys.stderr)
        img, label = example["image"], example["label"]
        output.write({"__key__": "%08d" % i, "jpg": img.convert("RGB"), "cls": label})
    output.close()
    time_taken = time.time() - now
    print(f"Wrote {i+1} train examples in {time_taken // 3600} hours.")

    opat = os.path.join(output_dir, "imagenet-val-%06d.tar")
    output = wds.ShardWriter(opat, maxcount=max_val_samples_per_shard)
    dataset = load_dataset("imagenet-1k", streaming=True, split="validation", use_auth_token=True)
    now = time.time()
    for i, example in enumerate(dataset):
        if i % max_val_samples_per_shard == 0:
            print(i, file=sys.stderr)
        img, label = example["image"], example["label"]
        output.write({"__key__": "%08d" % i, "jpg": img.convert("RGB"), "cls": label})
    output.close()
    time_taken = time.time() - now
    print(f"Wrote {i+1} val examples in {time_taken // 60} min.")

def convert_video_dataset_to_video_wds(output_dir, dataset, name, max_train_samples_per_shard, max_val_samples_per_shard, stack_frames, keep_every):
    assert not os.path.exists(os.path.join(output_dir, f"{name}-train-000000.tar"))
    assert not os.path.exists(os.path.join(output_dir, f"{name}-val-000000.tar"))

    opat = os.path.join(output_dir, f"{name}-train-%06d.tar")
    output = wds.ShardWriter(opat, maxcount=max_train_samples_per_shard)
    now = time.time()
    frame_counter = 0
    for video_i, (video, label) in enumerate(tqdm.tqdm(dataset)):
        if video_i == int(len(dataset) * 0.9):
            print("Switching to val set", file=sys.stderr)
            output.close()
            output = wds.ShardWriter(opat.replace("train", "val"), maxcount=max_val_samples_per_shard)
            frame_counter = 0

        for frame_i in range(0, video.shape[0]-stack_frames*keep_every, stack_frames*keep_every):
            if frame_counter % max_train_samples_per_shard == 0:
                print(frame_counter, file=sys.stderr)

            images = []
            for seq_frame_i in range(0, stack_frames*keep_every, keep_every):
                images.append(video[frame_i+seq_frame_i])

            # cat images side by side (in the width dimension)
            # img = np.concatenate(images, axis=1)
            # video_seq = [Image.fromarray(img) for img in images]
            video_npy = np.stack(images, axis=0) # T x H x W x C
            video_torch = torch.from_numpy(video_npy).permute(0, 3, 1, 2) # C x T x H x W
            # video_torch = video_torch.float()

            output.write({"__key__": "%08d" % frame_counter, "sequence.pth": video_torch, "cls": label})
            frame_counter += 1
    output.close()
    time_taken = time.time() - now
    print(f"Wrote {frame_counter+1} train examples in {time_taken // 3600} hours.")

def convert_video_dataset_to_wds(output_dir, dataset, keep_every, name, max_train_samples_per_shard, max_val_samples_per_shard, stack_frames):
    assert not os.path.exists(os.path.join(output_dir, f"{name}-train-000000.tar"))
    assert not os.path.exists(os.path.join(output_dir, f"{name}-val-000000.tar"))

    opat = os.path.join(output_dir, f"{name}-train-%06d.tar")
    output = wds.ShardWriter(opat, maxcount=max_train_samples_per_shard)
    # dataset = load_dataset("imagenet-1k", streaming=True, split="train", use_auth_token=True)
    now = time.time()
    frame_counter = 0
    for i, (video, label) in enumerate(tqdm.tqdm(dataset)):
        if i == int(len(dataset) * 0.9):
            print("Switching to val set", file=sys.stderr)
            output.close()
            output = wds.ShardWriter(opat.replace("train", "val"), maxcount=max_val_samples_per_shard)
            frame_counter = 0

        for i in range(0, video.shape[0]-stack_frames, max(keep_every, stack_frames)):
            if frame_counter % max_train_samples_per_shard == 0:
                print(frame_counter, file=sys.stderr)

            images = []
            for j in range(stack_frames):
                images.append(video[i+j])

            # cat images side by side (in the width dimension)
            img = np.concatenate(images, axis=1)

            output.write({"__key__": "%08d" % frame_counter, "jpg": Image.fromarray(img), "cls": label})
            frame_counter += 1
    output.close()
    time_taken = time.time() - now
    print(f"Wrote {frame_counter+1} train examples in {time_taken // 3600} hours.")

    # opat = os.path.join(output_dir, "imagenet-val-%06d.tar")
    # output = wds.ShardWriter(opat, maxcount=max_val_samples_per_shard)
    # dataset = load_dataset("imagenet-1k", streaming=True, split="validation", use_auth_token=True)
    # now = time.time()
    # for i, example in enumerate(dataset):
    #     if i % max_val_samples_per_shard == 0:
    #         print(i, file=sys.stderr)
    #     img, label = example["image"], example["label"]
    #     output.write({"__key__": "%08d" % i, "jpg": img.convert("RGB"), "cls": label})
    # output.close()
    # time_taken = time.time() - now
    # print(f"Wrote {i+1} val examples in {time_taken // 60} min.")


if __name__ == "__main__":
    # create parase object
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_train_samples_per_shard", type=int, default=4000)
    parser.add_argument("--max_val_samples_per_shard", type=int, default=1000)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--keep_every", type=int, default=2)
    parser.add_argument("--stack_frames", type=int, default=1)
    args = parser.parse_args()

    if args.dataset == "dmlab":
        dataset = DmlabDataset("../teco/dmlab/train/")
    elif args.dataset == "minecraft":
        dataset = MinecraftDataset("../teco/minecraft/train")
    elif args.dataset == "ucf101":
        dataset = UCF_101_Dataset("/mnt/data/vras/data/ucf_101/train")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    # convert_video_dataset_to_wds(args.output_dir, dataset, args.keep_every, args.dataset, args.max_train_samples_per_shard, args.max_val_samples_per_shard, args.stack_frames)
    convert_video_dataset_to_video_wds(args.output_dir, dataset, args.dataset, args.max_train_samples_per_shard, args.max_val_samples_per_shard, args.stack_frames, args.keep_every)

    # convert_imagenet_to_wds(args.output_dir, args.max_train_samples_per_shard, args.max_val_samples_per_shard)
