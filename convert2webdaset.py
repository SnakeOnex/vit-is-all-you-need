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

import webdataset as wds
# from datasets import load_dataset
from datasets import DmlabDataset


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

def convert_video_dataset_to_wds(output_dir, dataset, keep_every, name, max_train_samples_per_shard, max_val_samples_per_shard):
    assert not os.path.exists(os.path.join(output_dir, f"{name}-train-000000.tar"))
    assert not os.path.exists(os.path.join(output_dir, f"{name}-val-000000.tar"))

    opat = os.path.join(output_dir, "imagenet-train-%06d.tar")
    output = wds.ShardWriter(opat, maxcount=max_train_samples_per_shard)
    # dataset = load_dataset("imagenet-1k", streaming=True, split="train", use_auth_token=True)
    now = time.time()
    for i, (video, actions) in enumerate(dataset):
        print(f"original video shape: {video.shape}")
        video = video[::keep_every]
        actions = actions[::keep_every]
        print(f"video shape: {video.shape}")
        exit(0)


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


if __name__ == "__main__":
    # create parase object
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_train_samples_per_shard", type=int, default=4000)
    parser.add_argument("--max_val_samples_per_shard", type=int, default=1000)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--keep_every", type=int, default=1)
    args = parser.parse_args()

    if args.dataset == "dmlab":
        dataset = DmlabDataset("../teco/dmlab/train/")
        print(dataset)

    exit(0)

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    convert_video_dataset_to_wds(args.output_dir, dataset, keep_every, name, args.max_train_samples_per_shard, args.max_val_samples_per_shard)

    # convert_imagenet_to_wds(args.output_dir, args.max_train_samples_per_shard, args.max_val_samples_per_shard)
