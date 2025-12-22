
```zsh
conda create -n qwen_lora python=3.10
conda activate qwen_lora
pip install transformers
pip install ipython ipykernel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install modelscope
pip install av==14.3.0
pip install qwen-omni-utils==0.0.4
pip install accelerate
pip install flash_attn-2.7.1.post4+cu11torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install ipywidgets
conda install datasets
pip install peft
pip install 'qwen-omni-utils[decord]' -U
```