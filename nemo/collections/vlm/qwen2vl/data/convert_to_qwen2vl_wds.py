import json
import os
import pickle
from argparse import ArgumentParser

import webdataset as wds
from torchvision.io import encode_jpeg
from tqdm import tqdm
from webdataset.writer import add_handlers, default_handlers, imageencoder

os.environ["FORCE_QWENVL_VIDEO_READER"] = 'torchvision'
import numpy as np
import torch
from vision_process import fetch_image, fetch_video


def convert(dataset_dir, json_name, max_count=10000, mediate_path=''):
    """
    Here we provide an example to convert llava-pretrain dataset to webdataset
    """

    # Paths to the dataset files
    json_file = os.path.join(dataset_dir, json_name)
    output = os.path.join(dataset_dir, 'wds')

    if not os.path.exists(output):
        os.mkdir(output)

    # Load data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # custom webdataset ShardWriter Encoder
    add_handlers(default_handlers, "jpgs", lambda data: pickle.dumps([np.array(d) for d in data]))
    add_handlers(
        default_handlers, "videos", lambda data: pickle.dumps([[np.array(d) for d in video] for video in data])
    )

    has_idx = None
    with wds.ShardWriter(os.path.join(output, 'pretrain-%d.tar'), maxcount=max_count) as shard_writer:
        for idx, entry in enumerate(tqdm(data)):
            # NOTE: read a dataset in sharegpt format
            image_datas = []
            if 'image' in entry:
                pop_item = entry.pop('image')
            elif 'images' in entry:
                pop_item = entry.pop('images')
            else:
                pop_item = []

            if not isinstance(pop_item, list):
                pop_item = [pop_item]
            for image in pop_item:
                file_path = os.path.normpath(os.path.join(dataset_dir, mediate_path, image))
                image_datas.append(fetch_image({"image": file_path}))

            video_datas = []
            if 'video' in entry:
                pop_item = entry.pop('video')
            elif 'videos' in entry:
                pop_item = entry.pop('videos')
            else:
                pop_item = []

            if not isinstance(pop_item, list):
                pop_item = [pop_item]
            for video in pop_item:
                file_path = os.path.normpath(os.path.join(dataset_dir, mediate_path, video))
                fvideo = fetch_video({"video": file_path})
                video_datas.append(fvideo)

            if has_idx is None:
                has_idx = 'id' in entry
            assert has_idx == ('id' in entry), "All entries should either all contain idx or not."
            if 'conversations' in entry:
                conv = json.dumps(entry['conversations']).encode("utf-8")
            elif 'messages' in entry:
                conv = json.dumps(entry['messages']).encode("utf-8")
            else:
                conv = None
            assert conv is not None, "No conversation texts"

            sample = {
                "__key__": entry.pop('id', str(idx)),
                "jpgs": image_datas,
                'videos': video_datas,
                "json": conv,
            }
            shard_writer.write(sample)

    print(f"Dataset successfully converted to wds")
    return output


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--dataset-root', required=True, type=str)
    argparser.add_argument('--json', default='dataset.json', type=str)
    argparser.add_argument('--max-samples-per-tar', default=10000, type=float)
    argparser.add_argument('--mediate-path', default='', type=str)
    args = argparser.parse_args()

    output_dir = convert(
        args.dataset_root, args.json, max_count=args.max_samples_per_tar, mediate_path=args.mediate_path
    )
    print(f"convert done")
