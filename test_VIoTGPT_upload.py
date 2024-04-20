# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# coding: utf-8
import os
import gc
import gradio as gr
gr.close_all()
import torch
import numpy as np
import argparse
import inspect
import json
import jsonlines
import re
import uuid
from PIL import Image

from VIoTGPT_Vision_nodemo import (
    FaceRecognition,
    PlateRecognition,
    PersonReid,
    GaitRecognition,
    VehicleReid,
    FSDetect,
    CrowdCounting,
    HumanAction,
    HumanPose,
    SceneRecognition,
    AnomalyDetection
)
from VIoTGPT_Vision_nodemo import PREFIX, FORMAT_INSTRUCTIONS, SUFFIX

from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory

from llama_model import LlamaModel
from lora_model import LoraModel

os.makedirs('image', exist_ok=True)

def cut_dialogue_history(history_memory, keep_last_n_words=500):
    if history_memory is None or len(history_memory) == 0:
        return history_memory
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"history_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    paragraphs = history_memory.split('\n')
    last_n_tokens = n_tokens
    while last_n_tokens >= keep_last_n_words:
        last_n_tokens -= len(paragraphs[0].split(' '))
        paragraphs = paragraphs[1:]
    return '\n' + '\n'.join(paragraphs)

class ConversationBot:
    def __init__(self, load_dict):
        print(f"Initializing VideoNetGPT, load_dict={load_dict}")
        self.models = {}
        # Load Basic Foundation Models
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)
        # Load Template Foundation Models
        for class_name, module in globals().items():
            if getattr(module, 'template_model', False):
                template_required_names = {k for k in inspect.signature(module.__init__).parameters.keys() if
                                           k != 'self'}
                loaded_names = set([type(e).__name__ for e in self.models.values()])
                if template_required_names.issubset(loaded_names):
                    self.models[class_name] = globals()[class_name](
                        **{name: self.models[name] for name in template_required_names})
        print(f"All the Available Functions: {self.models}")
        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith('inference'):
                    func = getattr(instance, e)
                    self.tools.append(Tool(name=func.name, description=func.description, func=func))
        if args.lora_path == "":
            self.llm = LlamaModel(args.model_path)
        else:
            self.llm = LoraModel(base_name_or_path=args.model_path, model_name_or_path=args.lora_path, #load_8bit=False)
                                 load_8bit=True)
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': PREFIX, 'format_instructions': FORMAT_INSTRUCTIONS,
                          'suffix': SUFFIX},
            handle_parsing_errors=True)

    def run_text(self, text):
        self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=1000)
        res = self.agent({"input": text.strip()})
        inter_logs = []
        for i in range(len(res['intermediate_steps'])):
            inter_logs.append(res['intermediate_steps'][i][0].log)
        res['output'] = res['output'].replace("\\", "/")
        response = re.sub('(image/[-\w]*.png)', lambda m: f'![](file={m.group(0)})*{m.group(0)}*', res['output'])
        print(f"\nProcessed run_text, Input text: {text}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return inter_logs, response

    def run_image(self, image, txt):
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        print("======>Auto Resize Image...")
        img = Image.open(image)
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        width_new = int(np.round(width_new / 64.0)) * 64
        height_new = int(np.round(height_new / 64.0)) * 64
        img = img.resize((width_new, height_new))
        img = img.convert('RGB')
        img.save(image_filename, "PNG")
        print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
        Human_prompt = f'\nHuman: provide a figure named {image_filename}. ' \
                       f'You can use one or several tools to finish following tasks, rather than directly imagine. ' \
                       f'Especially, you will never use nonexistent tools. ' \
                       f'Once you have the final answer, do tell me in the format of "Final Answer: [your response here]". \n'
        AI_prompt = f'Received. I will tell you in the format of "Final Answer: [your response here]"'
        self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
        print(f"\nProcessed run_image, Input image: {image_filename}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return f'{txt} {image_filename} '


if __name__ == '__main__':
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str,
                        default="FaceRecognition_cuda:0")
    parser.add_argument('--model_path', type=str,
                        default="./models/7B_hf/")
    parser.add_argument('--lora_path', type=str,
                        default="",
                        required=False,
                        help='tool-llama lora model path')
    parser.add_argument('--output_path', type=str, default="", required=True,
                        help='Preprocessed tool data output path.')
    parser.add_argument('--query_path', type=str, default="",
                        help='query_path.')
    parser.add_argument('--query_data_path', type=str, default="",
                        help='query_path.')

    args = parser.parse_args()
    load_dict = {e.split('_')[0].strip(): e.split('_')[1].strip() for e in args.load.split(',')}
    print("load_dict", load_dict)
    data_dicts = json.load(open(args.query_path, "r"))
    len_query_file = len(data_dicts)
    out_file = jsonlines.open(args.output_path, "w")
    out_file._flush = True
    for i in range(len_query_file):
        out_list = []
        data_dict = data_dicts[i]
        query = data_dict["query"]
        img_path = "{}{}".format(args.query_data_path, data_dict["img_path"][1:])
        print(i, img_path)
        bot = ConversationBot(load_dict=load_dict)
        bot.memory.clear()
        bot.run_image(img_path, "Processed run_image, Input image: ")
        print(i, "query:", query)
        try:
            inter_logs, response = bot.run_text(query)
            print("inter_logs", inter_logs)
            print("response", response)
        except ValueError:
            response = "==========ValueError: Could not parse LLM output=========="
            inter_logs = "==========ValueError: Could not parse LLM output=========="
        except Exception as e:
            response = "==========Unknown error of VNGPT=========="
            inter_logs = "==========Unknown error of VNGPT=========="
            print(i, "error: ", e)
        except:
            response = "==========Unknown error of VNGPT=========="
            inter_logs = "==========Unknown error of VNGPT=========="
        dict = {
            "image_name_GT": img_path,
            "id": query,
            "chains": inter_logs,
            "result": response
        }
        print(dict)
        out_file.write(dict)
        del bot
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
