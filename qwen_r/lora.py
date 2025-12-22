import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,7"
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
        video_id = sample["id"]
        video_path = os.path.join(self.video_root, f"{video_id}.mp4")

        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": sample["conversations"][0]["value"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["conversations"][1]["value"]},
                ],
            },
        ]

        return {"conversation": conversation}

class QwenOmniDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def __call__(self, features):
        texts = []
        videos = []
        audios = []
        labels_list = []

        for f in features:
            conversation = f["conversation"]

            # ---------- 1. 拼完整 prompt ----------
            full_text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
            )

            # ---------- 2. 构造 labels（后缀 assistant） ----------
            assistant_text = conversation[-1]["content"][0]["text"]

            full_ids = self.tokenizer(
                full_text,
                add_special_tokens=False,
            )["input_ids"]

            assistant_ids = self.tokenizer(
                assistant_text,
                add_special_tokens=False,
            )["input_ids"]

            labels = [-100] * len(full_ids)
            labels[-len(assistant_ids):] = assistant_ids

            texts.append(full_text)
            labels_list.append(labels)

            # ---------- 3. 多模态 ----------
            for msg in conversation:
                if msg["role"] == "user":
                    for ele in msg["content"]:
                        if ele.get("type") == "video":
                            ele["fps"] = 0.5
                            ele["max_frames"] = 50
                            ele["min_pixels"] = 64 * 28 * 28
                            ele["max_pixels"] = 64 * 28 * 28

            audios_, _, videos_ = process_mm_info(
                conversation, use_audio_in_video=True
            )

            videos.append(videos_[0] if videos_ else None)
            audios.append(audios_[0] if audios_ else None)

        # ---------- 4. 一次性 processor ----------
        batch = self.processor(
            text=texts,
            videos=videos,
            audio=audios,
            padding=True,
            return_tensors="pt",
            use_audio_in_video=True,
        )

        print(batch["video_grid_thw"].shape) 
        print(batch["pixel_values_videos"].shape) 
        for k, v in batch.items(): 
            if isinstance(v, torch.Tensor): 
                print(k, v.shape, v.numel() * v.element_size() / 1024**3, "GB")


        # ---------- 5. pad labels ----------
        max_len = batch["input_ids"].shape[1]
        padded_labels = []

        for lab in labels_list:
            padded = lab + [-100] * (max_len - len(lab))
            padded_labels.append(padded)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

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

# model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
#     model_path,
#     dtype=torch.bfloat16,
#     device_map="balanced",
#     trust_remote_code=True,
#     use_safetensors=True
# )
# model = model.thinker

video_processor = FixedResQwen2VLVideoProcessor.from_pretrained(model_path)

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
    bias="none"
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
    fp16=False,
    num_train_epochs=2, # 5 -> 2
    logging_steps=5,
    load_best_model_at_end=False,
)

data_collator = QwenOmniDataCollator(processor)

trainer = Trainer(
    model=model,
    # model=model.thinker,
    args=args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer.train()

