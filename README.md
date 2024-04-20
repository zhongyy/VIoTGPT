# VIoTGPT
Codes, models, and VIoT-Tool dataset.

### VIoTGPT 
Download the pre-trained LLama-2-7b and Vicuna-7b, then download the LoRA weights from the following links.
1. [LLama-lora](https://drive.google.com/drive/folders/10kY7uuAH4XqVKTALK3jjy4VnUYzTaVZJ?usp=sharing).
2. [Vicuna-lora](https://drive.google.com/drive/folders/10eF7fMXktVJZ30kdyVjuPUluAnJsw5o6?usp=sharing).

### VToT-Tool dataset
1. [training](https://drive.google.com/file/d/10ZTkusnE5OjzUgi7JsD6JeY-nkRluSx9/view?usp=sharing). 
2. [testing](https://drive.google.com/file/d/11A_m3ORWmJEoD8YGf7LrmLW_7-TLtvQZ/view?usp=sharing), copy it to "./VIoT_tool"

### Environment
```
torch==2.0.1
transformers==4.31.0
deepspeed==0.10.0
langchain==0.0.101
gradio==3.23.0
```
### Train and Test
1. Train
```
cd train
train_vicuna.sh
```
2. Test
```
test_vicuna.sh 
```
