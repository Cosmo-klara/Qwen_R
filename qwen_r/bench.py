import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from modelscope import snapshot_download
from modelscope import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
import re
from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor
from qwen_omni_utils import process_mm_info
import json
from torch.utils.data import Dataset
from transformers.image_utils import SizeDict

def replace_time_tokens_with_percentage(text, time_map, duration):

    if not time_map or duration is None:
        return text

    def repl(match):
        token = match.group(0)
        if token not in time_map:
            return token
        t = time_map[token]
        pct = t / duration * 100.0
        return f"{pct:.1f}%"

    return re.sub(r"<s\d+>|<e\d+>", repl, text)


class OmniVideoConversationDataset(Dataset):
    def __init__(self, json_path: str, video_root: str):
        with open(json_path, "r") as f:
            raw_data = json.load(f)

        self.video_root = video_root
        self.samples = []

        for item in raw_data:
            video_id = item["id"]
            video_path = os.path.join(video_root, f"{video_id}.mp4")

            convs = item["conversations"]
            meta = item.get("meta", {})
            duration = meta.get("duration", None)
            time_map = meta.get("token", {})
            

            # 遍历 human / gpt 成对
            for i in range(0, len(convs) - 1, 2):
                if convs[i]["from"] != "human" or convs[i + 1]["from"] != "gpt":
                    continue

                self.samples.append({
                    "video_path": video_path,
                    "question": convs[i]["value"],
                    "answer": convs[i + 1]["value"],
                    "duration": duration,
                    "time_map": time_map,
                })

    def __len__(self):
        return len(self.samples)


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
        s = self.samples[idx]

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
                    {"type": "video", "video": s["video_path"]},
                    {"type": "text", "text": s["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": s["answer"]},
                ],
            },
        ]

        return {
            "conversation": conversation,
            "duration": s["duration"],
            "time_map": s["time_map"],
            }


class QwenOmniDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def __call__(self, features):
        texts = []
        videos = []
        audios = []

        for f in features:
            conversation = f["conversation"]

            for msg in conversation:
                if msg["role"] in ("user", "assistant"):
                    for ele in msg["content"]:
                        if ele.get("type") == "text":
                            ele["text"] = replace_time_tokens_with_percentage(
                                ele["text"],
                                f["time_map"],
                                f["duration"],
                            )

            # ---------- 1. 拼完整 prompt ----------
            full_text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
            )

            # ---------- 2. 构造 labels（后缀 assistant） ----------
            texts.append(full_text)

            # ---------- 3. 多模态 ----------
            for msg in conversation:
                if msg["role"] == "user":
                    for ele in msg["content"]:
                        if ele.get("type") == "video":
                            ele["fps"] = 0.5
                            ele["max_frames"] = 50
                            # ele["min_pixels"] = 64 * 28 * 28
                            # ele["max_pixels"] = 64 * 28 * 28

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
            use_audio_in_video=True
        )


        labels = batch["input_ids"].clone()
        labels[:] = -100

        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_id = self.tokenizer.convert_tokens_to_ids("assistant")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        for b in range(labels.size(0)):
            input_ids = batch["input_ids"][b]

            start = None
            for i in range(len(input_ids) - 1):
                if input_ids[i] == im_start_id and input_ids[i + 1] == assistant_id:
                    start = i + 3
                    break

            if start is None:
                raise RuntimeError("No <|im_start|> assistant found")

            end = None
            for i in range(start, len(input_ids)):
                if input_ids[i] == im_end_id:
                    end = i
                    break

            if end is None:
                end = len(input_ids)

            labels[b, start:end] = input_ids[start:end]

        batch["labels"] = labels

        # print(batch["labels"])

        # print(batch["video_grid_thw"].shape) 

        # print(batch["pixel_values_videos"].shape) 
        # for k, v in batch.items(): 
        #     if isinstance(v, torch.Tensor): 
        #         print(k, v.shape, v.numel() * v.element_size() / 1024**3, "GB")

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
    device_map="auto",
    trust_remote_code=True,
    use_safetensors=True
)

class FixedResQwen2VLVideoProcessor(Qwen2VLVideoProcessor):
    def _preprocess(
        self, videos, do_resize=True, size=None, interpolation=None, **kwargs
    ):
        # 固定分辨率
        fixed_size = SizeDict(height=224, width=224)
        for i, video in enumerate(videos):
            videos[i] = self.resize(video, size=fixed_size, interpolation=interpolation)
        return super()._preprocess(videos, do_resize=False, size=fixed_size, interpolation=interpolation, **kwargs)
    
video_processor = FixedResQwen2VLVideoProcessor.from_pretrained(model_path)

processor = Qwen2_5OmniProcessor.from_pretrained(
    model_path,
    video_processor=video_processor,
)


# 配置LoRA
config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    # task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)


for name, param in model.named_parameters():
    if (
        "audio_tower" in name
        or "visual" in name
    ):
        param.requires_grad = False

model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False

model.print_trainable_parameters()

# 检查模型是否在训练模式
model.train()
print(f"Model is in training mode: {model.training}")

batch_size = 1

args = TrainingArguments(
    output_dir="./r_models",
    remove_unused_columns=False,
    eval_strategy="no",
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=2,
    bf16=True,
    fp16=False,
    num_train_epochs=2,
    logging_steps=5,
    load_best_model_at_end=False,
)

data_collator = QwenOmniDataCollator(processor)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer.train()


