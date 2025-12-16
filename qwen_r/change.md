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

