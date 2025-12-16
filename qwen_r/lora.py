import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from modelscope import snapshot_download
from modelscope import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor
from transformers.video_utils import VideoMetadata
from typing import Optional

import json
from torch.utils.data import Dataset

class OmniVideoConversationDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        video_root: str,
    ):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.video_root = video_root

    def __len__(self):
        return len(self.data)

    def _build_text(self, conversations):
        messages = []
        for turn in conversations:
            if turn["from"] == "human":
                role = "user"
            elif turn["from"] == "gpt":
                role = "assistant"
            else:
                continue

            messages.append({
                "role": role,
                "content": turn["value"]
            })

        return messages

    def __getitem__(self, idx):
        sample = self.data[idx]
        messages = self._build_text(sample["conversations"])
        video_id = sample["id"]
        video_path = os.path.join(self.video_root, f"{video_id}.mp4")

        return {
            "text": messages,
            "videos": [video_path],
        }

train_dataset = OmniVideoConversationDataset(
    json_path="../../LongVALE/data/longvale-sft-bp-7k.json",
    video_root="../../LongVALE/raw_videos_train"
)


class FixedFrameQwen2VLVideoProcessor(Qwen2VLVideoProcessor):
    def __init__(self, num_frames=100, **kwargs):
        super().__init__(**kwargs)
        self.fixed_num_frames = num_frames

    def sample_frames(
        self,
        metadata: VideoMetadata,
        temporal_patch_size: Optional[int] = None,
        **kwargs,
    ):
        temporal_patch_size = temporal_patch_size or self.temporal_patch_size

        # 对齐 temporal_patch_size（Qwen2.5 默认 = 2）
        num_frames = round(self.fixed_num_frames / temporal_patch_size) * temporal_patch_size
        num_frames = min(num_frames, metadata.total_num_frames)

        indices = torch.linspace(
            0,
            metadata.total_num_frames - 1,
            steps=num_frames,
        ).long()

        return indices

model_path = snapshot_download(
    'Qwen/Qwen2.5-Omni-3B',
    cache_dir="../../Qwen/cache/modelscope"
)

model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda:1",
    trust_remote_code=True,
    use_safetensors=True
)

video_processor = FixedFrameQwen2VLVideoProcessor.from_pretrained(model_path)
processor = Qwen2_5OmniProcessor.from_pretrained(
    model_path,
    video_processor=video_processor,
)

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"],
    # target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

batch_size = 32

args = TrainingArguments(
    output_dir="./r_models",
    remove_unused_columns=False,
    eval_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    # per_device_eval_batch_size=batch_size,
    bf16=True,
    num_train_epochs=3, # 5 -> 2
    logging_steps=10,
    save_steps=100,
    load_best_model_at_end=False,
    label_names=["labels"],
)

trainer = Trainer(
    # model=model.thinker,
    model=model,
    args=args,
    train_dataset=train_dataset,
    processing_class=processor
)

