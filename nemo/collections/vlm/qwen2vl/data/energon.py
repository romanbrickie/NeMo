# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import dataclasses
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from megatron.energon import Batch, DefaultTaskEncoder, batch_list, batch_pad_stack
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.vqa import VQASample
from megatron.energon.task_encoder.cooking import Cooker, basic_sample_keys
from PIL import Image
from torchvision.io import decode_jpeg

from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.collections.vlm.qwen2vl.data.multimodal_tokens import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    PAD_TOKEN_INDEX,
    SPECIAL_TOKEN_MAP,
    VIDEO_TOKEN_INDEX,
    VISION_END_TOKEN_INDEX,
)
from nemo.collections.vlm.qwen2vl.data.preloaded import process_vision
from nemo.utils import logging


@dataclass
class ChatMLSample(Sample):
    # __key__: str
    # __subflavors__: Dict
    imgs: List[Image.Image]
    videos: List[torch.Tensor | list[Image.Image]]
    conversation: str  # JSON string of GPT-format conversations


# Type for encoded sample
@dataclass
class Qwen2VLTaskSample:
    __key__: str
    __subflavors__: Dict

    imgs: List[torch.Tensor]  # (c, h, w)
    videos: List[torch.Tensor]  # (c, h, w)

    image_thw_grids: List[torch.Tensor]
    video_thw_grids: List[torch.Tensor]
    image_input_mask: torch.Tensor
    video_input_mask: torch.Tensor
    text: torch.Tensor
    target: torch.Tensor


# Typing for the resulting batch data after encode_batch()
@dataclass
class Qwen2VLTaskBatch(Batch):
    __keys__: List[str]
    __subflavors__: List[Dict]
    # (num_tiles, c, h, w)
    pixel_values: torch.Tensor
    pixel_values_videos: torch.Tensor
    image_grid_thw: torch.Tensor
    video_grid_thw: torch.Tensor
    image_input_mask: torch.Tensor
    video_input_mask: torch.Tensor
    # (n, seq_len)
    input_ids: torch.Tensor
    # (n, seq_len)
    labels: torch.Tensor
    loss_mask: torch.Tensor


def convert_to_qwen2vl_content(user_input: str, image_pattern: str = '<image>', video_pattern: str = '<video>'):
    """
    Split user input into format Qwen2VL tokenizer accepts.
    """
    pattern = r"({image}|{video})".format(image=image_pattern, video=video_pattern)
    contents = []
    cur = 0
    mm_idx = defaultdict(int)
    for matched in re.finditer(pattern, user_input):
        start, end = matched.span()
        if start > cur:
            contents.append({"type": "text", "text": user_input[cur:start].strip(' ')})

        contents.append(
            {
                "type": matched.string[start:end][1:-1],
                matched.string[start:end][1:-1]: str(mm_idx[matched.string[start:end][1:-1]]),
            }
        )

        cur = end
        mm_idx[matched.string[start:end][1:-1]] += 1

    if cur < len(user_input):
        contents.append({"type": "text", "text": user_input[cur : len(user_input)].strip(' ')})

    return contents


def cook_chatml_sample(sample: dict) -> ChatMLSample:
    imgs = sample.get('jpgs', None)
    if imgs:
        imgs = pickle.loads(imgs)
        if isinstance(imgs, list) and len(imgs) > 0:
            imgs = [Image.fromarray(d) for d in imgs]
        else:
            imgs = None
    videos = sample.get('videos', None)
    if videos:
        videos = pickle.loads(videos)
        if isinstance(videos, list) and len(videos) > 0:
            videos = [[d for d in video] for video in videos]
        else:
            videos = None
    if "<image>" in sample['json'] and imgs is None:
        log.warning("<image> in conversation text but no image data")
    if "<video>" in sample['json'] and videos is None:
        log.warning("<video> in conversation text but no video data")

    chat_sample = ChatMLSample(
        **basic_sample_keys(sample),
        imgs=imgs,
        videos=videos,
        conversation=sample['json'],
    )
    return chat_sample


class Qwen2VLTaskEncoder(DefaultTaskEncoder[ChatMLSample, Qwen2VLTaskSample, Qwen2VLTaskBatch, dict]):
    """A simple task encoder for captioning."""

    cookers = [
        Cooker(cook_chatml_sample),
    ]

    def __init__(
        self,
        tokenizer,
        image_processor,
        # data_paths: str | List[str],
        # data_config,
        # global_batch_size: int = 8,
        # micro_batch_size: int = 1,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        patch_size: int = 14,
        max_padding_length: int = 4096,
    ):
        # Specify the batch_type for default batching (batching is performed here "manually" by
        # overwriting the `batch` method)
        super().__init__()

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        # self.data_paths = data_paths

        # self.data_config = data_config
        self.seq_length = max_padding_length
        # self.gbs = global_batch_size
        # self.mbs = micro_batch_size

        self.temporal_patch_size = temporal_patch_size
        self.merge_size = spatial_merge_size
        self.patch_size = patch_size

        self.seq_len = max_padding_length
        self.image_token_id, self.video_token_id = self.tokenizer.encode(['<|image_pad|>', '<|video_pad|>'])

    def encode_sample(self, sample: ChatMLSample):

        # NOTE: flatten all images
        #     Input of process_vision:
        #      sample.imgs in list[PIL.Image.Image]
        #      sample.videos in list[Tensor]
        #   see definition of ChatMLSample or convert_to_qwen2vl_wds.py, structure is restored by Cooker
        processed_vision = process_vision(self.image_processor, sample.imgs, sample.videos)
        image_thw_grids = processed_vision['image_grid_thw']
        video_thw_grids = processed_vision['video_grid_thw']
        flattened_imgs = processed_vision['image_inputs']
        flattened_videos = processed_vision['video_inputs']

        # NOTE: generate qwen2vl conversations
        conversation = (
            json.loads(sample.conversation) if isinstance(sample.conversation, (str, bytes)) else sample.conversation
        )

        _from_system_ = 'from' in conversation[0]
        role_key = 'from' if 'from' in conversation[0] else 'role'
        content_key = 'value' if 'from' in conversation[0] else 'content'

        # NOTE: assume the conversation format is: [System]? (User Assistant)+
        converted_conversation = []
        if len(conversation) % 2 == 0:
            # Default Prompt
            converted_conversation.append({'role': 'system', 'content': 'You are a helpful assistant.'})
        else:
            converted_conversation.append({'role': 'system', 'content': conversation[0][content_key]})
            conversation = conversation[1:]

        if _from_system_:  # ['conversations':[{'from':'human', 'value':[]}, {'from':'gpt', 'value':[]}]
            EXPECTED_ROLE = ['human', 'gpt']
            for turn_idx, turn in enumerate(conversation):
                role = turn[role_key]
                if role != EXPECTED_ROLE[turn_idx % len(EXPECTED_ROLE)]:
                    logging.warning(
                        f"Expect conversation organized in order: [sys] human gpt human gpt..., but got role '{role}' in turn {turn_idx}"
                    )
                content = turn[content_key]

                if role == 'human':
                    role = 'user'
                    content = convert_to_qwen2vl_content(content)
                elif role == 'gpt':
                    role = 'assistant'

                converted_conversation.append({'role': role, 'content': content})
        else:  # ['messages':[{'role':'user', 'content':[]}, {'role':'assistant', 'content':[]}]
            EXPECTED_ROLE = ['user', 'assistant']
            for turn_idx, turn in enumerate(conversation):
                role = turn[role_key]
                if role != EXPECTED_ROLE[turn_idx % len(EXPECTED_ROLE)]:
                    logging.warning(
                        f"Expect conversation organized in order: [sys] user assistant user assistant..., but got role '{role}' in turn {turn_idx}"
                    )
                content = turn[content_key]

                if role == 'user':
                    content = convert_to_qwen2vl_content(content)

                converted_conversation.append({'role': role, 'content': content})
        conversation = converted_conversation

        # NOTE: we need to mask all system/user input tokens and assistant generation prefix tokens
        input_ids = self.tokenizer.apply_chat_template(conversation, tokenize=True, return_tensors="np")[0]
        target = input_ids.copy()

        system_prompt_prefix = len(self.tokenizer.apply_chat_template([conversation[0]], tokenize=True))
        assistant_generation_prefix = 3
        pad_token_id = self.tokenizer.pad_token_id

        target[:system_prompt_prefix] = pad_token_id
        offset = system_prompt_prefix
        for turn_idx, turn in enumerate(conversation[1:]):
            turn_tokens = self.tokenizer.apply_chat_template([turn], tokenize=True, return_tensors="np")[0]
            turn_content = turn_tokens[system_prompt_prefix:]
            n_tokens = len(turn_content)
            if (target[offset : offset + n_tokens] != turn_content).any():
                logging.warning("Encode Error")

            if turn['role'] == 'user':
                target[offset : offset + n_tokens] = pad_token_id
            elif turn['role'] == 'assistant':
                target[offset : offset + assistant_generation_prefix] = pad_token_id
            offset += n_tokens

        # NOTE: expand image_pad & video_pad
        merge_length = self.merge_size**2
        image_token_id, video_token_id = self.image_token_id, self.video_token_id

        image_token_indices = np.where(input_ids == image_token_id)[0]
        if image_token_indices is not None and image_thw_grids is not None:
            assert len(image_token_indices) == len(
                image_thw_grids
            ), f"With {len(image_thw_grids)} images in the sample, but {len(image_token_indices)} image placeholders!"
        video_token_indices = np.where(input_ids == video_token_id)[0]
        if video_token_indices is not None and video_thw_grids is not None:
            assert len(video_token_indices) == len(
                video_thw_grids
            ), f"With {len(video_thw_grids)} videos in the sample, but {len(video_token_indices)} video placeholders!"
        if image_thw_grids is not None and video_thw_grids is not None:
            image_thw_grids, video_thw_grids = np.array(image_thw_grids, dtype=np.int64), np.array(
                video_thw_grids, dtype=np.int64
            )

            target_length = (
                input_ids.shape[0]
                - image_thw_grids.shape[0]
                + image_thw_grids.prod(axis=-1).sum() // merge_length
                - video_thw_grids.shape[0]
                + video_thw_grids.prod(axis=-1).sum() // merge_length
            )
        elif image_thw_grids is not None:
            image_thw_grids = np.array(image_thw_grids, dtype=np.int64)

            target_length = (
                input_ids.shape[0] - image_thw_grids.shape[0] + image_thw_grids.prod(axis=-1).sum() // merge_length
            )
        elif video_thw_grids is not None:
            video_thw_grids = np.array(video_thw_grids, dtype=np.int64)

            target_length = (
                input_ids.shape[0] - video_thw_grids.shape[0] + video_thw_grids.prod(axis=-1).sum() // merge_length
            )
        else:
            target_length = input_ids.shape[0]

        if target_length > self.seq_len:
            logging.warning(f"Long sequence with length {target_length} found, dropped...")
        final_input_ids = np.zeros(target_length, dtype=input_ids.dtype)
        final_input_masks = final_input_ids.copy()

        image_idx, video_idx = 0, 0
        indices = np.sort(np.concatenate([image_token_indices, video_token_indices]))

        cur_x, cur_y = 0, 0
        for idx in indices:
            token_id = input_ids[idx]
            if token_id == image_token_id:
                size = image_thw_grids[image_idx].prod() // merge_length
                image_idx += 1
            elif token_id == video_token_id:
                size = video_thw_grids[video_idx].prod() // merge_length
                video_idx += 1
            # NOTE:
            # input_ids[cur_x:idx] -> final_input_ids[cur_y:cur_y + idx - cur_x]
            # input_ids[idx] -> final_input_ids[cur_y + idx - cur_x: cur_y + idx - cur_x + size]
            final_input_ids[cur_y : cur_y + idx - cur_x] = input_ids[cur_x:idx]
            final_input_masks[cur_y : cur_y + idx - cur_x] = target[cur_x:idx]
            cur_y += idx - cur_x
            final_input_ids[cur_y : cur_y + size] = token_id
            final_input_masks[cur_y : cur_y + size] = pad_token_id
            cur_y += size
            cur_x = idx + 1

        if cur_x < len(input_ids):
            final_input_ids[cur_y:] = input_ids[cur_x:]
            final_input_masks[cur_y:] = target[cur_x:]

        target = np.roll(final_input_masks, shift=-1)
        target[-1] = pad_token_id

        if (target == pad_token_id).all():
            log.warning("Sample with all masked label, dropped.")

        image_input_mask = torch.from_numpy(final_input_ids == image_token_id)
        video_input_mask = torch.from_numpy(final_input_ids == video_token_id)
        # collect data
        return Qwen2VLTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            imgs=flattened_imgs['pixel_values'] if flattened_imgs else [],
            videos=flattened_videos['pixel_values_videos'] if flattened_videos else [],
            image_thw_grids=image_thw_grids if flattened_imgs else [],
            video_thw_grids=video_thw_grids if flattened_videos else [],
            image_input_mask=image_input_mask,
            video_input_mask=video_input_mask,
            text=torch.from_numpy(final_input_ids),
            target=torch.from_numpy(target),
        )

    def batch(self, samples: List[Qwen2VLTaskSample]) -> Qwen2VLTaskBatch:
        # Stack images to [num_tiles, c, h, w]. If there are no images (text-only), then use a dummy image.
        imgs, image_thw_grids = [], []
        for s in samples:
            if len(s.imgs) > 0:
                s_imgs = [img for img in s.imgs.unsqueeze(0)]
                if len(s_imgs) > 0:
                    cat_imgs = torch.cat([img for img in s_imgs])
                    imgs.append(cat_imgs)
                else:
                    imgs.append(
                        torch.empty(
                            [0, 3 * self.temporal_patch_size * self.patch_size * self.patch_size], dtype=torch.float32
                        )
                    )
            if len(s.image_thw_grids) > 0:
                s_image_thw_grids = [thw_grids for thw_grids in s.image_thw_grids]
                if len(s_image_thw_grids) > 0:
                    image_thw_grids.extend(s_image_thw_grids)
                else:
                    image_thw_grids = torch.cat((image_thw_grids, torch.empty([0, 3], dtype=torch.long)))
        # Stack videos to [num_tiles, c, h, w]. If there are no videos (text-only), then use a dummy video.
        videos, video_thw_grids = [], []
        for s in samples:
            if len(s.videos) > 0:
                s_videos = [video for video in s.videos.unsqueeze(0)]
                if len(s_videos) > 0:
                    cat_videos = torch.cat([video for video in s_videos])
                    videos.append(cat_videos)
                else:
                    videos = torch.cat(
                        torch.empty(
                            [0, 3 * self.temporal_patch_size * self.patch_size * self.patch_size], dtype=torch.float32
                        )
                    )
            if len(s.video_thw_grids) > 0:
                s_video_thw_grids = [thw_grids for thw_grids in s.video_thw_grids]
                if len(s_video_thw_grids) > 0:
                    video_thw_grids.extend(s_video_thw_grids)
                    # assert s_video_thw_grids.prod(dim=-1).sum() == s_videos.shape[0]
                else:
                    video_thw_grids = torch.cat(torch.empty([0, 3], dtype=torch.long))

        # If the user hasn't defined a target sequence length, then use the max along the sample lengths.
        max_seq_len = max(len(s.text) for s in samples)
        if max_seq_len > self.seq_len:
            log.warning("max sequence length larger than passed parameter")

        text_mat = np.full((len(samples), max_seq_len), self.tokenizer.pad_token_id, dtype=np.int64)
        # +1 to accommodate shift to left by one later.
        target_mat = np.full((len(samples), max_seq_len), self.tokenizer.pad_token_id, dtype=np.int64)

        image_input_masks = np.zeros_like(text_mat, dtype=bool)
        video_input_masks = np.zeros_like(text_mat, dtype=bool)
        for i, s in enumerate(samples):
            # If the sample/target length exceeds the target sequence length, then truncate.
            text_len = min(max_seq_len, len(s.text))
            target_len = min(max_seq_len, len(s.target))

            text_mat[i, :text_len] = np.array(s.text)[:text_len]
            # NOTE: we should assert user input sequence will not be truncated
            if s.image_input_mask is not None:
                image_input_masks[i, :text_len] = np.array(s.image_input_mask)[:text_len]
            if s.video_input_mask is not None:
                video_input_masks[i, :text_len] = np.array(s.video_input_mask)[:text_len]
            target_mat[i, :target_len] = np.array(s.target)[:target_len]

        tokens = torch.from_numpy(text_mat)
        tokens[tokens == self.image_token_id] = IMAGE_TOKEN_INDEX
        tokens[tokens == self.video_token_id] = VIDEO_TOKEN_INDEX
        tokens[tokens == PAD_TOKEN_INDEX] = 0

        labels = torch.from_numpy(target_mat)
        labels[labels == PAD_TOKEN_INDEX] = IGNORE_INDEX

        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            data=tokens,
            eod_token=self.tokenizer.eos_token_id,
            eod_mask_loss=False,
            reset_attention_mask=False,
            reset_position_ids=False,
            # eod_mask_loss=self.data_config.eod_mask_loss,
            # reset_attention_mask=self.data_config.reset_attention_mask,
            # reset_position_ids=self.data_config.reset_position_ids,
        )

        loss_mask[labels < 0] = 0.0

        batch = Qwen2VLTaskBatch(
            __keys__=[s.__key__ for s in samples],
            __subflavors__=[s.__subflavors__ for s in samples],
            pixel_values=torch.vstack(imgs) if len(imgs) > 0 else None,
            pixel_values_videos=torch.vstack(videos) if len(videos) > 0 else None,
            image_grid_thw=image_thw_grids if len(image_thw_grids) > 0 else None,
            video_grid_thw=video_thw_grids if len(video_thw_grids) > 0 else None,
            image_input_mask=torch.from_numpy(image_input_masks),
            video_input_mask=torch.from_numpy(video_input_masks),
            input_ids=tokens,
            labels=labels,
            loss_mask=loss_mask,
        )
        return batch

    def encode_batch(self, batch: Qwen2VLTaskBatch) -> dict:
        raw = dataclasses.asdict(batch)
        del raw["__subflavors__"]
        return raw
