发现 transformers 在模型的 forward 中直接调用了 self.loss_function

```py
    loss = None
    if labels is not None:
        loss = self.loss_function(
            logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size
        )
```
 
往上定位 self.loss_funciton 的实现时，发现其定义在父类 PreTrainedModel 中( src/transformers/modeling_utils.py 内)：

```py
class Qwen2_5OmniPreTrainedModel(PreTrainedModel):
class Qwen2_5OmniPreTrainedModelForConditionalGeneration(Qwen2_5OmniPreTrainedModel):
class Qwen2_5OmniThinkerForConditionalGeneration(Qwen2_5OmniPreTrainedModelForConditionalGeneration, GenerationMixin):
```

```py
    @property
    def loss_function(self):
        if hasattr(self, "_loss_function"):
            return self._loss_function

        loss_type = getattr(self, "loss_type", None)

        if loss_type is None or loss_type not in LOSS_MAPPING:
            logger.warning_once(
                f"`loss_type={loss_type}` was set in the config but it is unrecognized. "
                f"Using the default loss: `ForCausalLMLoss`."
            )
            loss_type = "ForCausalLM"
        return LOSS_MAPPING[loss_type]

    @loss_function.setter
    def loss_function(self, value):
        self._loss_function = value
```

首先会检测是否包含 _loss_function 属性，如果存在该属性，则直接使用 self._loss_function 自定义定义的损失函数；

然后没有自定义损失函数的话就通过 LOSS_MAPPING 字典映射，找 loss_type 对应的损失函数；如果 loss_type 不在 LOSS_MAPPING 中或者 loss_type 为 None，默认使用 “ForCausalLM” 对应的损失函数

在 LOSS_MAPPING 中找 "ForCausalLM": ForCausalLMLoss, 实现如下：

```py
def ForCausalLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: torch.Tensor | None = None,
    ignore_index: int = -100,
    shift_labels: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()

    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(logits.device)
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss
```

其中 fixed_cross_entropy 实现如下：

```py
def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: torch.Tensor | None = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        # just in case users pass an int for num_items_in_batch, which could be the case for custom trainer
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss
```

可以看到没有做优化之类的；或许可以参照 [cce](https://github.com/apple/ml-cross-entropy) 或者 [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) 里面提到的减少训练显存的使用；

后面发现 transformers 已经支持了 Liger-Kernel，需要在 TrainingArguments 中启用，还有就是可以开 torch_compile=True ；等跑完一轮试试


