import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from modelscope import snapshot_download
from modelscope import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration


model_path = snapshot_download(
    'Qwen/Qwen2.5-Omni-3B',
    cache_dir="./cache/modelscope"
)

train_dataset = load_dataset("huggingface/cats-image")

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda:1",
    trust_remote_code=True,
    enable_audio_output=False,
    use_safetensors=True
)

# model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     device_map="cuda:1",
#     trust_remote_code=True,
#     enable_audio_output=False,
#     use_safetensors=True
# )

# 这个要改
processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

batch_size = 128

args = TrainingArguments(
    output_dir="./r_models",
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-3,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    bf16=True,
    num_train_epochs=2, # 5 -> 2
    logging_steps=10,
    load_best_model_at_end=True,
    label_names=["labels"],
)

trainer = Trainer(
    model=model.thinker,
    args=args,
    train_dataset=train_dataset,
    processing_class=image_processor,
    data_collator=collate_fn,
)


