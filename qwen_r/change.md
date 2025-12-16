### exp2

固定均匀采样 50 帧 + 真实时间重映射()

Qwen2.5-omni 使用 TM-RoPE，也就是说依赖于：(token_index, physical_time)。其中 physical_time 就来自视频元信息。

每个 video token 都隐式绑定一个时间长度：

second_per_grid_ts = [ video_processor.temporal_patch_size / fps ] * num_video_grids

second_per_grid_ts = [self.video_processor.temporal_patch_size / fps] * len(video_grid_thw)

len(video_grid_thw)： video time-chunks 的数量

也就是：

“这个 token 覆盖了现实世界中多少秒”

TM-RoPE 用它来算 真实时间轴上的相位偏移。

```py
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"

from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor
from transformers.video_utils import VideoMetadata
import torch
from typing import Optional

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

import soundfile as sf
from modelscope import snapshot_download
from modelscope import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

model_dir = snapshot_download(
    'Qwen/Qwen2.5-Omni-3B',
    cache_dir="../../Qwen/cache/modelscope"
)

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_dir,
    device_map="cuda:1",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model.disable_talker()

video_processor = FixedFrameQwen2VLVideoProcessor.from_pretrained(model_dir)
processor = Qwen2_5OmniProcessor.from_pretrained(
    model_dir,
    video_processor=video_processor,
)

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
            {"type": "video", "video": "./test.mp4"},
            {"type": "text", "text": "Could you detail events during different time segments? Format strictly:\nFrom xx to xx, event1.\nFrom xx to xx, event2.\n..."}
        ],
    },
]

USE_AUDIO_IN_VIDEO = True

text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(model.device).to(model.dtype)
text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)

text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text)
```