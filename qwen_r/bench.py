import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import copy
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
            self.raw_data = json.load(f)

        self.video_root = video_root

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        item = self.raw_data[idx]

        video_id = item["id"]
        video_path = os.path.join(self.video_root, f"{video_id}.mp4")
        audio_path = video_path.replace(".mp4", ".wav")

        duration = item.get("meta", {}).get("duration", None)
        time_map = item.get("meta", {}).get("token", {})

        return {
            "video_path": video_path,
            "audio_path": audio_path,
            "conversations": copy.deepcopy(item["conversations"]),
            "duration": duration,
            "time_map": time_map,
        }


class QwenOmniDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def _replace_time_tokens(self, conversations, time_map, duration):
        if not time_map or duration is None:
            return conversations

        def repl(match):
            token = match.group(0)
            if token not in time_map:
                return token
            pct = time_map[token] / duration * 100
            return f"{pct:.1f}%"

        for turn in conversations:
            turn["value"] = re.sub(r"<s\d+>|<e\d+>", repl, turn["value"])
        return conversations
    
    def _split_rounds(self, conversations):
        rounds = []
        cur = []
        for turn in conversations:
            cur.append(turn)
            if turn["from"] == "gpt":
                rounds.append(cur)
                cur = []
        return rounds

    def _truncate_by_round(
        self,
        base_chat,          # system + video/audio
        rounds,             # [[h,g], [h,g], ...]
        max_length
    ):
        while True:
            chat = copy.deepcopy(base_chat)

            for r in rounds:
                for t in r:
                    role = "user" if t["from"] == "human" else "assistant"
                    chat.append({
                        "role": role,
                        "content": [{"type": "text", "text": t["value"]}],
                    })

            prompt = self.processor.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=False
            )

            token_len = len(self.tokenizer(prompt).input_ids)

            if token_len <= max_length or len(rounds) <= 1:
                return chat

            # ❗ 删除最早的一整轮
            rounds = rounds[1:]

    def _build_conversation(self, sample):
        conversations = self._replace_time_tokens(
            sample["conversations"],
            sample["time_map"],
            sample["duration"],
        )

        base_chat = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": sample["video_path"], "fps": 0.5, "max_frames": 50},
                    {"type": "audio", "audio": sample["audio_path"]},
                ],
            },
        ]

        rounds = self._split_rounds(conversations)

        chat = self._truncate_by_round(
            base_chat=base_chat,
            rounds=rounds,
            max_length=self.tokenizer.model_max_length,
        )

        return chat


    def _build_labels(self, input_ids):
        labels = input_ids.clone()
        labels[:] = -100

        im_start = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        assistant = self.tokenizer.convert_tokens_to_ids("assistant")

        i = 0
        while i < len(input_ids) - 1:
            if input_ids[i] == im_start and input_ids[i + 1] == assistant:
                j = i + 3  # skip <|im_start|> assistant \n
                while j < len(input_ids) and input_ids[j] != im_end:
                    labels[j] = input_ids[j]
                    j += 1
                i = j
            else:
                i += 1
        return labels

    
    def __call__(self, features):
        texts, videos, audios = [], [], []

        for sample in features:
            conversation = self._build_conversation(sample)

            prompt = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(prompt)

            # video_path = sample["video_path"]
            # if video_path not in self.video_cache:
            #     audios_, _, videos_ = process_mm_info(
            #         conversation, use_audio_in_video=False
            #     )
            #     self.video_cache[video_path] = {
            #         "video": videos_[0],
            #         "audio": audios_[0],
            #     }

            # videos.append(self.video_cache[video_path]["video"])
            # audios.append(self.video_cache[video_path]["audio"])

            audios_, _, videos_ = process_mm_info(
                    conversation, use_audio_in_video=False
                )

            videos.append(videos_[0] if videos_ else None)
            audios.append(audios_[0] if audios_ else None)

        batch = self.processor(
            text=texts,
            videos=videos,
            audio=audios,
            padding=True,
            return_tensors="pt",
            use_audio_in_video=False,
        )

        labels = torch.stack([
            self._build_labels(ids)
            for ids in batch["input_ids"]
        ])

        batch["labels"] = labels
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
    gradient_accumulation_steps=1,
    bf16=True,
    fp16=False,
    num_train_epochs=1,
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


