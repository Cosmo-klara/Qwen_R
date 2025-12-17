### exp2

固定均匀采样 100 帧 + 真实时间重映射()

Qwen2.5-omni 使用 TM-RoPE，也就是说依赖于：(token_index, physical_time)。其中 physical_time 就来自视频元信息。

每个 video token 都隐式绑定一个时间长度：

second_per_grid_ts = [ video_processor.temporal_patch_size / fps ] * num_video_grids

second_per_grid_ts = [self.video_processor.temporal_patch_size / fps] * len(video_grid_thw)

len(video_grid_thw)： video time-chunks 的数量

也就是：

“这个 token 覆盖了现实世界中多少秒”

TM-RoPE 用它来算 真实时间轴上的相位偏移。

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import torch

from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from modelscope import snapshot_download
from modelscope import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor
from transformers.video_utils import VideoMetadata
from typing import Optional
from qwen_omni_utils import process_mm_info
import json
from torch.utils.data import Dataset
from transformers.image_utils import SizeDict

class FixedResQwen2VLVideoProcessor(Qwen2VLVideoProcessor):
    def _preprocess(
        self, videos, do_resize=True, size=None, interpolation=None, **kwargs
    ):
        # 固定分辨率
        fixed_size = SizeDict(height=224, width=224)
        for i, video in enumerate(videos):
            videos[i] = self.resize(video, size=fixed_size, interpolation=interpolation)
        return super()._preprocess(videos, do_resize=False, size=fixed_size, interpolation=interpolation, **kwargs)

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

def build_prompt(messages):
    prompt = ""
    for m in messages:
        if m["role"] == "user":
            prompt += f"<human>{m['content']}</human>"
        elif m["role"] == "assistant":
            prompt += f"<gpt>{m['content']}</gpt>"
    return prompt

class QwenOmniDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        texts = [build_prompt(f["text"]) for f in features]
        print(texts[0])

        # 视频路径列表
        videos = [f["videos"][0] if f.get("videos") else None for f in features]

        batch = self.processor(
            text=texts,
            videos=videos,
            padding=True,
            return_tensors="pt",
            use_audio_in_video=True,
        )

        print(batch["video_grid_thw"].shape)
        print(batch["pixel_values_videos"].shape)
        
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape, v.numel() * v.element_size() / 1024**3, "GB")

        return batch

train_dataset = OmniVideoConversationDataset(
    json_path="../../LongVALE/data/longvale-sft-bp-7k.json",
    video_root="../../LongVALE/raw_videos_train/video_train_7240/"
)

model_path = snapshot_download(
    'Qwen/Qwen2.5-Omni-3B',
    cache_dir="../../Qwen/cache/modelscope"
)

model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="balanced",
    trust_remote_code=True,
    use_safetensors=True
)

video_processor = FixedResQwen2VLVideoProcessor.from_pretrained(model_path)
video_processor.do_sample_frames = True
video_processor.fps = 2.0

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

for name, param in model.named_parameters():
    if (
        "audio_tower" in name
        or "visual" in name
    ):
        param.requires_grad = False
model.gradient_checkpointing_enable()
model.config.use_cache = False

model.print_trainable_parameters()

batch_size = 1

args = TrainingArguments(
    output_dir="./r_models",
    remove_unused_columns=False,
    eval_strategy="no",
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=2,
    # per_device_eval_batch_size=batch_size,
    bf16=True,
    num_train_epochs=2, # 5 -> 2
    logging_steps=5,
    load_best_model_at_end=False,
    label_names=["labels"],
)

data_collator = QwenOmniDataCollator(processor)

trainer = Trainer(
    model=model,
    # model=model.thinker,
    args=args,
    train_dataset=train_dataset,
    processing_class=processor,
    data_collator=data_collator,
)
```