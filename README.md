# VIoTGPT
Codes, models, and VIoT-Tool dataset.

### VIoTGPT 
1. [LLama-lora]().
2. [Vicuna-lora]().

### VToT-Tool dataset
1. [training](https://drive.google.com/file/d/10ZTkusnE5OjzUgi7JsD6JeY-nkRluSx9/view?usp=sharing). 
2. [testing](). 

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
