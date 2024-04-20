# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# coding: utf-8
import os
import sys

import random
import torch
import cv2
import shutil

import numpy as np

sys.path.append('./tools/')
from posec3d.ntu_pose_extraction import extract_frame, detection_inference, ntu_det_postproc, pose_inference
import mmcv
from posec3d.action_recognition import inference_pytorch
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.utils import setup_multi_processes
from mmcv import Config

sys.path.append('./tools/reid-test/')
sys.path.append('./tools/reid-test/demo/')
from fastreid.config import get_cfg
from predictor import FeatureExtractionDemo
import torch.nn.functional as F

sys.path.append('./tools/fire-smoke-detection/')
from pathlib import Path
import queue
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import (non_max_suppression, scale_coords, xyxy2xywh, plot_one_box)

sys.path.append('./tools/CrowdCounting-P2PNet-main/')
from config_crowd import Crowdcfg
import torchvision.transforms as standard_transforms
from Crowdengine import *
from Crowdmodels import build_model

sys.path.append('./tools/plate_recognition/rpnet/')
from PIL import Image, ImageDraw, ImageFont
from demo_plate import fh02

sys.path.append('./tools/PEL4VAD-master-past/')
from configs import build_config
from model import XModel
from dataset import *
from video_features.extract_i3d import *
from infer import infer_func
from utils_pel4 import setup_seed
from log import get_logger

sys.path.append('./tools/Gait-recognition/')
from track import *
from segment import *
from recognise import *

sys.path.append('./tools/place/')
import pandas as pd
from torchvision import transforms
import wideresnet
from tqdm import *

PREFIX = """VIoTGPT is designed to help multi-modal video surveillance analysis on VIoT. 
    VIoTGPT cannot directly read images or videos, but it has a series of visual tools to accomplish different monitoring. 
    Each image will have a file name formed as "image/xxx.png" and each video will have a file name formed as "video/xxx.mp4". 
    VIoTGPT can invoke different tools to indirectly understand the picture and the video indirectly. 
    VIoTGPT is very strict about filenames and will never fake nonexistent files. 
    VIoTGPT is able to use tools in a sequence, and is loyal to the tool observation outputs rather than faking the image content and image file name. 

    VIoTGPT has a video database, which now consists of videos named "video/xxx.mp4", 
    like "video/Wuhan.mp4", video/Beijing.mp4", 
    "video/Shanghai.mp4", "video/Guangzhou.mp4", 
    "video/Nanjing.mp4", "video/Kunming.mp4" and so on.

    TOOLS:
    ------

    VIoTGPT has access to the following tools:"""

FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```
When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No.
{ai_prefix}: Final Answer: [your response here]
```
"""

SUFFIX = """You are very strict to the filename correctness and will never fake a file name if it does not exist.
You will remember to provide the image file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Since VIoTGPT is a text language model, VIoTGPT must use tools to observe images rather than imagination.
The thoughts and observations are only visible for VIoTGPT, VIoTGPT should remember to repeat important information in the final response for Human. 
Let's think step by step. {agent_scratchpad} 
"""

os.makedirs('image', exist_ok=True)

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func
    return decorator

def visualize(image, faces, return_msg, thickness=2):
    input = image.copy()
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print(
                'Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(
                    idx, face[0], face[1], face[2], face[3], face[-1]))
            coords = face[:-1].astype(np.int32)
            cv2.rectangle(input, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0),
                          thickness)
            cv2.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv2.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv2.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv2.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv2.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv2.putText(input, return_msg, (1, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), thickness)
    return input

class SceneRecognition:
    def __init__(self, device):
        print("Initializing Scene Recognition")
        self.device = device
        self.classes, self.labels_IO, self.labels_attribute, self.W_attribute = self.__load_labels()
        self.features_blobs = []
        self.model = self.__load_model().to(self.device)
        params = list(self.model.parameters())
        self.weight_softmax = params[-2].data
        self.weight_softmax[self.weight_softmax<0] = 0
        self.ratio = 0.1
        self.trasform = pd.read_csv('./tools/place/transform.txt',
                                    header=None, index_col=0).to_dict()[1]
    def __recursion_change_bn(self, module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.track_running_stats = 1
        else:
            for i, (name, module1) in enumerate(module._modules.items()):
                module1 = self.__recursion_change_bn(module1)
        return module
    def __load_labels(self):
        file_name_category = './tools/place/categories_places365.txt'
        classes = list()
        with open(file_name_category) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])
        classes = tuple(classes)
        # indoor and outdoor relevant
        file_name_IO = './tools/place/IO_places365.txt'
        with open(file_name_IO) as f:
            lines = f.readlines()
            labels_IO = []
            for line in lines:
                items = line.rstrip().split()
                labels_IO.append(int(items[-1]) - 1)
        labels_IO = np.array(labels_IO)
        # scene attribute relevant
        file_name_attribute = './tools/place/labels_sunattribute.txt'
        with open(file_name_attribute) as f:
            lines = f.readlines()
            labels_attribute = [item.rstrip() for item in lines]
        file_name_W = './tools/place/W_sceneattribute_wideresnet18.npy'
        W_attribute = np.load(file_name_W)
        return classes, labels_IO, labels_attribute, W_attribute

    def __returnTF(self):
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return tf

    def __load_model(self):
        # this model has a last conv feature map as 14x14
        model_file = './tools/place/wideresnet18_places365.pth.tar'
        model = wideresnet.resnet18(num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
        for i, (name, module) in enumerate(model._modules.items()):
            module = self.__recursion_change_bn(model)
        model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)
        model.eval()
        return model
    @prompts(name="Recognize the Scene in the Video",
             description="useful when you want to determine the general semantic class of the scene in the video. "
                         "The input to this tool should be a string, representing the video_path. ")
    def inference(self, inputs):
        video_path = inputs
        video_path = video_path.split('/')[0] + '/Anomaly_' + video_path.split('/')[-1]
        video_cap = cv2.VideoCapture(video_path)
        nframes = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, img = video_cap.read()
        scene_idx_count = [0 for i in range(365)]
        if nframes > 50000:
            return 'other'
        for i in tqdm(range(nframes)):
            if not ret:
                break
            img = Image.fromarray(img)
            input_img = self.__returnTF()(img).unsqueeze(0).to(self.device)

            # forward pass
            with torch.no_grad():
                logit = self.model.forward(input_img)
                h_x = F.softmax(logit, 1).data.squeeze()
                idx = torch.max(h_x, 0)[1]
                scene_idx_count[idx] += 1
            ret, img = video_cap.read()
        scene = self.classes[scene_idx_count.index(max(scene_idx_count))]
        if scene in self.trasform.keys():
            scene = self.trasform[scene]
        else:
            scene = 'an unknown place'
        return scene

class AnomalyDetection:
    def __init__(self, device):
        print("Initializing Detect Anomaly")
        self.device = torch.device(device)
        self.cfg = build_config('xd')
        self.model = XModel(self.cfg)
        self.model.to(self.device)
        self.mode = 'infer'
        self.args_cli = {'feature_type': 'i3d', 'device': "cuda",
                         'video_paths': "./tools/UCF-Crime/shuffle/1/Abuse002_x264.mp4"}
        self.args_cli['device'] = device
        self.abnormal_dict = {'0': 'Normal', '1': 'Abuse', '2': 'Arrest', '3': 'Arson', '4': 'Assault', '5': 'Burglary',
                              '6': 'Explosion', '7': 'Fighting', '8': 'RoadAccidents', '9': 'Robbery', '10': 'Shooting',
                              '11': 'Shoplifting', '12': 'Stealing', '13': 'Vandalism'}

    def load_checkpoint(self, ckpt_path, logger):
        if os.path.isfile(ckpt_path):
            logger.info('loading pretrained checkpoint from {}.'.format(ckpt_path))
            weight_dict = torch.load(ckpt_path)
            model_dict = self.model.state_dict()
            for name, param in weight_dict.items():
                if 'module' in name:
                    name = '.'.join(name.split('.')[1:])
                if name in model_dict:
                    if param.size() == model_dict[name].size():
                        model_dict[name].copy_(param)
                    else:
                        logger.info('{} size mismatch: load {} given {}'.format(
                            name, param.size(), model_dict[name].size()))
                else:
                    logger.info('{} not found in model dict.'.format(name))
        else:
            logger.info('Not found pretrained checkpoint file.')

    @prompts(name="Detect Anomaly Condition on Video Scene",
             description="useful when you want to know whether there are anomalies that endanger public security in the video. "
                         "The input to this tool should be a comma separated string of two, representing the scene class and the video_path. ")
    def inference(self, inputs):
        scene_class, video_path = inputs.split(',')
        video_path = video_path.split('/')[0] + '/Anomaly_' + video_path.split('/')[-1]
        self.args_cli['video_paths'] = video_path
        if not os.path.exists(video_path):
            msg = "The file does not exist."
            return msg
        else:
            msg = 'All seemed normal in/on the {}.'.format(scene_class)
            return msg
        features = extract(self.args_cli)
        np.save('results/features.npy', np.array(features))
        logger = get_logger(self.cfg.logs_dir)
        print("debug:", self.cfg)
        setup_seed(self.cfg.seed)
        logger.info('Config:{}'.format(self.cfg.__dict__))
        if self.cfg.dataset == 'ucf-crime':
            train_data = UCFDataset(self.cfg, test_mode=False)
            test_data = UCFDataset(self.cfg, test_mode=True)
        elif self.cfg.dataset == 'xd-violence':
            train_data = XDataset(self.cfg, test_mode=False)
            test_data = XDataset(self.cfg, test_mode=True)
        elif self.cfg.dataset == 'shanghaiTech':
            train_data = SHDataset(self.cfg, test_mode=False)
            test_data = SHDataset(self.cfg, test_mode=True)
        else:
            raise RuntimeError("Do not support this dataset!")
        test_loader = DataLoader(test_data, batch_size=self.cfg.test_bs, shuffle=False,
                                 num_workers=self.cfg.workers, pin_memory=True)
        gt = np.load(self.cfg.gt)
        param = sum(p.numel() for p in self.model.parameters())
        logger.info('total params:{:.4f}M'.format(param / (1000 ** 2)))
        if self.mode == 'infer':
            logger.info('Test Mode')
            if self.cfg.ckpt_path is not None:
                self.load_checkpoint(self.cfg.ckpt_path, logger)
            else:
                logger.info('infer from random initialization')
            pred = infer_func(self.model, test_loader, gt, logger, self.cfg)
        else:
            raise RuntimeError('Invalid status!')
        pred = self.abnormal_dict[str(pred)]
        if pred == 'Normal':
            msg = 'All seemed normal in/on the {}.'.format(scene_class)
        else:
            msg = 'We found there is {} happening in/on the {}.'.format(pred, scene_class)

class GaitRecognition:
    def __init__(self, device):
        print("Initializing Gait Recognition")
        self.output_dir = "./OutputVideos/"
        os.makedirs(self.output_dir, exist_ok=True)
        self.device = device
    @prompts(name="Recognize the Person by Gait",
             description="useful when you want to know whether the person in the uploaded video appeared in the video. "
                         "The tool recognize people by gait, that is the way people walks or runs. "
                         "The input to this tool should be a comma separated string of two, representing the uploaded video_path and the target video_path. ")
    def inference(self, inputs):
        probe_video_path, gallery_video_path = inputs.split(',')
        print(probe_video_path, gallery_video_path)
        if 'mp4' not in probe_video_path:
            final_msg = 'mp4 {} does not exist.'.format(probe_video_path)
            return final_msg
        gallery_video_path = gallery_video_path.split('/')[0] + '/Gait_' + gallery_video_path.split('/')[-1]
        gallery_video = cv2.VideoCapture(gallery_video_path)
        probe_video = cv2.VideoCapture(probe_video_path)
        if not gallery_video.isOpened():
            final_msg = 'Video {} does not exist.'.format(gallery_video_path)
            return final_msg
        if not probe_video.isOpened():
            final_msg = 'Video {} does not exist.'.format(probe_video_path)
            return final_msg
        video_save_folder = './results'
        os.makedirs(video_save_folder, exist_ok=True)
        # tracking
        print('gallery tracking...')
        gallery_track_result = track(gallery_video_path, self.device)
        print('probe tracking...')
        probe_track_result = track(probe_video_path, self.device)

        print('gallery segmenting...')
        gallery_silhouette = seg(gallery_video_path, gallery_track_result, './GaitSilhouette/')
        print('probe segmenting...')
        probe_silhouette = seg(probe_video_path, probe_track_result, './GaitSilhouette/')
        # recognise
        gallery_feat = extract_sil(gallery_silhouette)
        probe1_feat = extract_sil(probe_silhouette)
        gallery_probe_result, scores = compare(probe1_feat, gallery_feat)
       # write the result back to the video
        print('getting result...')
        img_list = writeresult(gallery_probe_result, gallery_video_path, self.device)
        for i, img in enumerate(img_list):
            save_name = os.path.join(video_save_folder, f'result{i + 1}.jpg')
            cv2.imwrite(save_name, img)
        for key in scores:
            if scores[key] < 10.0:
                msg = 'According to the gait analysis, we have found this person. Please refer to {}. '.format(
                    save_name)
            else:
                msg = 'According to the gait analysis, we did not found this person.'
        return msg

class PlateRecognition:
    def __init__(self, device):
        self.device = device
        self.numClasses = 4
        self.img_size = (480, 480)
        self.resume_file = "./tools/plate_recognition/fh02.pth"
        self.provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁",
                          "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
        self.alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
        self.ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
        self.model = fh02()
        self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
        self.model.load_state_dict(torch.load(self.resume_file))
        self.model.cuda()
        self.model.eval()

    def preprocess(self, img):
        img = cv2.resize(img, self.img_size)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype('float32')
        img /= 255.0
        img = np.expand_dims(img, 0)
        img = torch.tensor(img)
        return img

    @prompts(name="Recognize the Vehicle by Plate Number",
             description="useful when you want to know whether the vehicle of given vehicle plate appeared in the video. "
                         "The tool recognize the vehicle by the plate number."
                         "The input to this tool should be a comma separated string of two, representing the vehicle plate and the video_path.")
    def inference(self, inputs):
        try:
            query_plate, video_path = inputs.split(',')
            video_path = video_path.split('/')[0] + '/PlateRecognition_' + video_path.split('/')[-1]
            video = cv2.VideoCapture(video_path)
            frame_cnt = 0
            if not video.isOpened():
                final_msg = 'Video {} does not exist.'.format(video_path)
                return final_msg
            while True:
                ret, frame = video.read()
                if frame is None:
                    final_msg = 'Vehicle plate {} does not appear in video {}'.format(query_plate, video_path)
                    return final_msg
                if ret is True:
                    frame_cnt += 1
                    img = self.preprocess(frame)
                    bbox, pred = self.model(img)
                    outputY = [el.data.cpu().numpy().tolist() for el in pred]
                    labelPred = [t[0].index(max(t[0])) for t in outputY]
                    [cx, cy, w, h] = bbox.data.cpu().numpy()[0].tolist()
                    lpn = self.provinces[labelPred[0]] + self.alphabets[labelPred[1]] + self.ads[labelPred[2]] + self.ads[
                        labelPred[3]] + self.ads[labelPred[4]] + self.ads[labelPred[5]] + self.ads[labelPred[6]]
                    if lpn == query_plate:
                        left_up = [(cx - w / 2) * frame.shape[1], (cy - h / 2) * frame.shape[0]]
                        right_down = [(cx + w / 2) * frame.shape[1], (cy + h / 2) * frame.shape[0]]
                        cv2.rectangle(frame, (int(left_up[0]), int(left_up[1])), (int(right_down[0]), int(right_down[1])),
                                      (0, 0, 255), 2)
                        pilImg = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(pilImg)
                        font = ImageFont.truetype("./tools/plate_recognition/rpnet/SimHei.ttf", 50, encoding='utf-8')
                        draw.text((int(left_up[0]), int(left_up[1]) - 40), lpn, (255, 0, 0),
                                  font=font)
                        cv2charimg = cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR)
                        cv2.imwrite("output/{}.jpg".format(frame_cnt), cv2charimg)
                        print("Image saved at output/{}.jpg".format(frame_cnt))
                        print("Query plate found at frame {}".format(frame_cnt))
                        final_msg = 'Video {} have this vehicle. The vehicle appeared in {} frames'.format(video_path, frame_cnt)
                        return final_msg
        except:
            return "Unkown Error in PlateRecognition."

class VehicleReid:
    def __init__(self, device):
        print(f"Initializing Vehicle Re-Identification")
        self.cfg = get_cfg()
        self.cfg.merge_from_file('./tools/reid-test/demo/Base-SBS.yml')
        self.cfg.merge_from_list(['MODEL.WEIGHTS', './tools/reid-test/demo/models/veri_sbs_R50-ibn.pth'])
        self.cfg.freeze()
        self.demo = FeatureExtractionDemo(self.cfg, parallel=False)
        self.threshold = 0.95

    def postprocess(self, features):
        features = F.normalize(features)
        features = features.cpu().data.numpy()
        return features

    @prompts(name="Recognize the Vehicle by Appearance",
             description="useful when you want to know whether the vehicle in the photo appeared in the video."
                         "The tool recognize the vehicle by appearance."
                         "The input to this tool should be a comma separated string of two, representing the image_path and the video_path.")
    def inference(self, inputs):
        try:
            final_msg = 'final_msg'
            query_path, gallery_path = inputs.split(',')
            if query_path is None:
                final_msg = 'Cannot find a query image in {}'.format(query_path)
                return final_msg
            query_img = cv2.imread(query_path)
            query_feat = self.demo.run_on_image(query_img)
            query_feat = self.postprocess(query_feat)
            gallery_path = gallery_path.split('/')[0] + '/VehicleReid_' + gallery_path.split('/')[-1]
            video = cv2.VideoCapture(gallery_path)
            frame_cnt = 0
            return_frame_id = []
            if not video.isOpened():
                final_msg = 'Video {} does not exist.'.format(gallery_path)
                return final_msg
            while True:
                ret, frame = video.read()
                if frame is None:
                    break
                if ret is True:
                    frame_cnt += 1
                    frame_feat = self.demo.run_on_image(frame)
                    frame_feat = self.postprocess(frame_feat)
                    cos_score = np.matmul(query_feat, frame_feat.T)
                    if cos_score >= self.threshold:
                        return_frame_id.append(frame_cnt)
            if len(return_frame_id) > 0 and len(return_frame_id) < 30:
                final_msg = 'Video {} have this vehicle. The vehicle appeared in {} frames, including Frame {}.'. \
                    format(gallery_path, len(return_frame_id), return_frame_id)
            elif len(return_frame_id) >= 30:
                final_msg = 'Video {} have this vehicle. The vehicle appeared in {} frames, including Frame {}, etc.'. \
                    format(gallery_path, len(return_frame_id), return_frame_id[:30])
            else:
                final_msg = 'Video {} does not have {}.'.format(gallery_path, 'this vehicle')
            print("final_msg: ", final_msg)
            return final_msg
        except:
            return "Unkown Error in VehicleReid."

class PersonReid:
    def __init__(self, device):
        print("Initializing Person Re-identification")
        self.cfg = get_cfg()
        self.cfg.merge_from_file('./tools/reid-test/demo/Base-SBS.yml')
        self.cfg.merge_from_list(['MODEL.WEIGHTS', './tools/reid-test/demo/models/market_sbs_R50-ibn.pth'])
        self.cfg.freeze()
        self.threshold = 0.975
        self.demo = FeatureExtractionDemo(self.cfg, parallel=False)

    def postprocess(self, features):
        features = F.normalize(features)
        features = features.cpu().data.numpy()
        return features

    @prompts(name="Recognize the Person by Appearance",
             description="useful when you want to know whether the person in the photo appeared in the video. "
                         "The tool recognize people by appearance, that is body shape and clothing. "
                         "The input to this tool should be a comma separated string of two, representing the image_path and the video_path.")
    def inference(self, inputs):
        try:
            query_path, gallery_path = inputs.split(',')
            if not os.path.exists(query_path) or query_path is None:
                final_msg = 'Cannot find a query image in {}'.format(query_path)
                return final_msg
            query_img = cv2.imread(query_path)
            query_feat = self.demo.run_on_image(query_img)
            query_feat = self.postprocess(query_feat)
            gallery_path = gallery_path.split('/')[0] + '/PersonReid_' + gallery_path.split('/')[-1]
            video = cv2.VideoCapture(gallery_path)
            frame_cnt = 0
            return_frame_id = []
            if not video.isOpened():
                final_msg = 'Video {} does not exist.'.format(gallery_path)
                return final_msg
            while True:
                ret, frame = video.read()
                if frame is None:
                    break
                if ret is True:
                    frame_cnt += 1
                    frame_feat = self.demo.run_on_image(frame)
                    frame_feat = self.postprocess(frame_feat)
                    cos_score = np.matmul(query_feat, frame_feat.T)
                    if cos_score >= self.threshold:
                        return_frame_id.append(frame_cnt)
            if len(return_frame_id) > 0 and len(return_frame_id) < 30:
                final_msg = 'Video {} have this identity. The person appeared in {} frames, including Frame {}.'. \
                    format(gallery_path, len(return_frame_id), return_frame_id)
            elif len(return_frame_id) >= 30:
                final_msg = 'Video {} have this identity. The person appeared in {} frames, including Frame {}, etc.'. \
                    format(gallery_path, len(return_frame_id), return_frame_id[:30])
            else:
                final_msg = 'Video {} does not have {}.'.format(gallery_path, 'this identity')
            print(final_msg)
            return final_msg
        except:
            return "Unkown Error in PersonReid"

class FaceRecognition:
    def __init__(self, device):
        print(f"Initializing FaceRecognition")
        self.detector = cv2.FaceDetectorYN.create(
            './tools/face_detection_yunet_2023mar.onnx',  # YuNet
            "",
            (320, 320),
            0.9,  # Filtering out faces of score < score_threshold
            0.3,  # Suppress bounding boxes of iou >= nms_threshold
            5000  # Keep top_k bounding boxes before NMS
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            './tools/face_recognition_sface_2021dec.onnx', "")
        self.cosine_similarity_threshold = 0.363
        self.l2_similarity_threshold = 1.128

    @prompts(name="Recognize the Face",
             description="useful when you want to know whether the faces in the photo appeared in the video. "
                         "The tool recognize people by face. "
                         "The input to this tool should be a comma separated string of two, representing the image_path and the video_path. ")
    def inference(self, inputs):
        try:
            image_path, video_path = inputs.split(',')
            # Detection of the uploaded image
            if image_path is not None:
                img1 = cv2.imread(cv2.samples.findFile(image_path))
            img1Width = int(img1.shape[1])
            img1Height = int(img1.shape[0])
            img1 = cv2.resize(img1, (img1Width, img1Height))
            self.detector.setInputSize((img1Width, img1Height))
            faces1 = self.detector.detect(img1)
            if faces1[1] is None:
                final_msg = 'Cannot find a face in {}'.format(image_path)
                return final_msg
            face1_align = self.recognizer.alignCrop(img1, faces1[1][0])
            face1_feature = self.recognizer.feature(face1_align)
            video_path = video_path.split('/')[0] + '/FaceRecognition_' + video_path.split('/')[-1]
            video = cv2.VideoCapture(video_path)
            frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.detector.setInputSize([frameWidth, frameHeight])
            frame_count = 0
            return_frame = []
            return_frame_id = []
            if video.isOpened():
                while True:
                    ret, frame = video.read()
                    if frame is None:
                        break
                    if ret == True:
                        frame_count = frame_count + 1
                        frame = cv2.resize(frame, (frameWidth, frameHeight))
                        faces2 = self.detector.detect(frame)
                        if faces2[1] is not None:
                            face2_align = self.recognizer.alignCrop(frame, faces2[1][0])
                            face2_feature = self.recognizer.feature(face2_align)
                            cosine_score = self.recognizer.match(face1_feature, face2_feature,
                                                                 cv2.FaceRecognizerSF_FR_COSINE)
                            if cosine_score >= self.cosine_similarity_threshold:
                                msg = 'the same identity'
                                return_msg = 'Frame {} have {}. Cosine Similarity: {}, threshold: {}.'. \
                                    format(frame_count, msg, cosine_score, self.cosine_similarity_threshold)
                                frame_result = visualize(frame, faces2, return_msg)
                                # cv2.imwrite('image/video_{}.jpg'.format(frame_count), frame_result)
                                return_frame.append(frame_result)
                                return_frame_id.append(frame_count)
            else:
                final_msg = 'Video {} does not exists.'.format(video_path)
                return final_msg
            video.release()
            if len(return_frame_id) > 0 and len(return_frame_id) < 30:
                final_msg = 'Video {} have {}. The person appeared in {} frames, including Frame {}.'. \
                    format(video_path, msg, len(return_frame_id), return_frame_id)
            elif len(return_frame_id) >= 30:
                final_msg = 'Video {} have {}. The person appeared in {} frames, including Frame {}, etc.'. \
                    format(video_path, msg, len(return_frame_id), return_frame_id[:30])
            else:
                final_msg = 'Video {} does not have {}.'.format(video_path, 'this identity')
            return final_msg
        except:
            return "Unknown error in FaceRecognition."

class HumanPose:
    def __init__(self, device):
        print(f"Initializing HumanPose to {device}")
        if 'cuda' in device:
            self.device = int(device.split('cuda:')[-1])

    @prompts(name="Detect the Human Pose",
             description="useful when you want to know the pose of human inside the video. "
                         "The input to this tool should be a string, representing the video_path.")
    def inference(self, video_path):
        try:
            if '/' in video_path:
                video_path = video_path.split('/')[0] + '/HumanPose_' + video_path.split('/')[-1]
            else:
                final_msg = 'Video {} does not exists.'.format(video_path)
                return final_msg
            print(video_path)
            if video_path is not None and os.path.exists(video_path):
                frame_paths = extract_frame(video_path)
                det_results = detection_inference(frame_paths, self.device)
                det_results = ntu_det_postproc(1, det_results)
                pose_results = pose_inference(frame_paths, det_results, self.device)
                anno = dict()
                anno['keypoint'] = pose_results[..., :2]
                anno['keypoint_score'] = pose_results[..., 2]
                anno['frame_dir'] = os.path.splitext(os.path.basename(video_path))[0]
                anno['img_shape'] = (1080, 1920)
                anno['original_shape'] = (1080, 1920)
                anno['total_frames'] = pose_results.shape[1]
                anno['label'] = 0
                total_anno = [anno]
                shutil.rmtree(os.path.dirname(frame_paths[0]))
                out_path = os.path.join('results', '{}.pkl'.format(video_path.split('/')[-1].split('.mp4')[0]))
                mmcv.dump(total_anno, out_path)
                final_msg = 'Pose estimation result of Video {} is saved in {}.'.format(video_path, out_path)
                print(final_msg)
            else:
                final_msg = 'Video {} does not exists.'.format(video_path)
            return final_msg, out_path
        except:
            return "Unkown Error in HumanPose", None

class HumanAction:
    def __init__(self, device='cpu'):
        print(f"Initializing HumanAction to {device}")
        self.checkpoint = './tools/posec3d/best_top1_acc_epoch_190.pth'
        self.config = './tools/posec3d/config.py'
        if 'cuda' in device:
            self.device = int(device.split('cuda:')[-1])

    @prompts(name="Recognize the Human Action Condition on Pose",
             description="useful when you want to know the action of human inside the video. "
                         "The input to this tool should be a string representing the path of the pose detection result. "
                         "Therefore, before using this tool, firstly you need to Detect the Human Pose."
                         )
    def inference(self, pose_path):
        try:
            if pose_path is not None and os.path.exists(pose_path):
                cfg = Config.fromfile(self.config)
                cfg.data.test.ann_file = pose_path
                cfg.data.ann_file_test = pose_path
                print("pose_out:", pose_path)
                setup_multi_processes(cfg)
                if cfg.get('cudnn_benchmark', False):
                    torch.backends.cudnn.benchmark = True
                cfg.data.test.test_mode = True
                cfg.setdefault('module_hooks', [])
                dataset = build_dataset(cfg.data.test, dict(test_mode=True))
                dataloader_setting = dict(
                    videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
                    workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
                    dist=False,
                    shuffle=False)
                dataloader_setting = dict(dataloader_setting,
                                          **cfg.data.get('test_dataloader', {}))
                data_loader = build_dataloader(dataset, **dataloader_setting)

                outputs = inference_pytorch(cfg, data_loader, self.checkpoint, [self.device])
                print(outputs)
                label = ["Pull Up", "Barbell Bench Press", "Barbell Bent-Over Row", "Barbell Upright Row",
                         "Barbell Deadlift", "Barbell Glute Bridge", "Barbell Stiff-leg Deadlift", "Barbell Back Squat",
                         "Barbell Front Squat", "Incline Barbell Bench Press", "Dcline Barbell Bench Press",
                         "Standing Barbell Curl", "Dumbbell Lateral Raise", "Laying Dumbbell Triceps Extension",
                         "Overhead Dumbbell Triceps Extension", "Alternating Dumbbell Curl", "Dumbbell fly",
                         "Seat Dumbbell Shoulder Press ", "Dumbbell Bent-over Row",
                         "Dumbbell Seated Bent-Over Lateral Raise", "Parallel Bars Leg Raise", "Seated Cable Row",
                         "Lat Pull-down", "Cable Front Raise", "Smith-machine Squat", "Smith Machine Shoulder Press ",
                         "Push-Up", "Kneeling Push-Up", "Release Push-Up", "Diamond Push-Up", "Burpee",
                         "Mountain Climbers", "Butt Kick", "Jumping Jack", "Jump Squat", "Standing Oblique Crunch",
                         "Standing Side Crunch", "Single Leg Stretch", "Double Leg Stretch", "Scissor Kick",
                         "Bent Knee Leg Raise", "Side kick（侧踢）", "Plank（平板支撑）", "Criss Cross/Bicycle crunch",
                         "Squat", "Crul-up/Crunch", "Sicilian Crunch（西西里卷腹）", "Reverse curls（反向卷腹）",
                         "Side Leg Lifts（侧卧举腿卷腹）", "Side crunch/Oblique crunch（侧身卷腹）",
                         " Simultaneous knee and abdominal raise/Keen lift crunch（仰卧屈膝两头起）",
                         "Straight leg toe touches（直腿触足卷腹）", "Elbow-to-knee crunch/Cross crunch（仰卧对角交替收膝）",
                         "Russian twist（俄罗斯转体）", "Glute bridge（臀桥）", "Single-Leg glute bridge（单腿臀桥）",
                         "Neck stretch（颈部拉伸）", "Stride stretch（跨步伸展 ）", "Abdominal stretch（腹部拉伸）",
                         "Calf stretch（小腿伸展）", "Lancelot stretch（举臂弓步伸展 ）", "Scarf stretch（肩胛伸展）",
                         "Chest stretch（胸部拉伸）", "Triceps stretch（手臂后侧拉伸）", "Quadriceps stretch（大腿前侧拉伸）",
                         "Hamstring stretch（坐姿开腿伸展/腿筋伸展）", "Side stretch（腹部侧拉）", "Single-Leg hopping（勾腿跳）",
                         "High knees（高抬腿）", "Leg swings（腿部摆动）", "Donkey kicks（驴踢）", "Half Moon Pose（Ardha Chandrasana）",
                         " Downward Dog (Adho Mukha Svanasana) ", "Handstand (Adho Mukha Vrksasana）", "Side Plank Pose（Vasisthasana）",
                         "Child’s Pose（Balasana）", "Garland Pose（Malasana）", "Dance Pose（Natarajasana）",
                         "Standing Wide-Legged Forward Fold Pose（Prasarita Padottanasana）", "Shoulderstand（Salamba sarvangasana）",
                         " Scorpion Pose（Vrschikasanae）", "Swaying Palm Tree Pose（Tiryaka Tadasana）", "Triangle Pose（Trikonasana）",
                         "Chair Pose（Utkatasana ）", "Standing Forward Bend Pose（Uttanasana）", "Warrior I（Virabhadrasana I）",
                         "Warrior II（Virabhadrasana II）", "Warrior III（Virabhadrasana III）"]
                max_k_preds = np.argsort(outputs, axis=1)[:, -1:][:, ::-1]
                result = label[max_k_preds[0][0]]
                final_msg = 'Action of Video {} is {}.'.format(pose_path, result)
            else:
                final_msg = 'Pose {} does not exists.'.format(pose_path)
            return final_msg, max_k_preds[0][0]
        except:
            return "Unkown Error in HumanAction", None

class ActionQuality:
    def __init__(self, device='cpu'):
        print(f"Initializing ActionQuality to {device}")
        self.checkpoint = './tools/posec3d/best_top1_acc_epoch_200.pth' # 176 classes
        self.config = './tools/posec3d/config_quality.py'
        if 'cuda' in device:
            self.device = int(device.split('cuda:')[-1])

    @prompts(name="Access the Quality of Actions Condition on both Pose and Action",
             description="useful when you want to know the quality of action inside the video."
                         "The input to this tool should be a comma separated string of two, representing the action_result and the pose_path."
             )
    def inference(self, inputs):
        try:
            action_result, pose_path = inputs.split(',')
            action_result, pose_path = action_result.strip(), pose_path.strip()
            if action_result is None or int(action_result) >= 88 or int(action_result) < 0:
                action_label = None
            else:
                action_label = int(action_result)
            if pose_path is not None and os.path.exists(pose_path):
                cfg = Config.fromfile(self.config)
                cfg.data.test.ann_file = pose_path
                cfg.data.ann_file_test = pose_path
                print("pose_out:", pose_path)
                setup_multi_processes(cfg)
                if cfg.get('cudnn_benchmark', False):
                    torch.backends.cudnn.benchmark = True
                cfg.data.test.test_mode = True

                cfg.setdefault('module_hooks', [])
                dataset = build_dataset(cfg.data.test, dict(test_mode=True))
                dataloader_setting = dict(
                    videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
                    workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
                    dist=False,
                    shuffle=False)
                dataloader_setting = dict(dataloader_setting,
                                          **cfg.data.get('test_dataloader', {}))
                data_loader = build_dataloader(dataset, **dataloader_setting)

                outputs = inference_pytorch(cfg, data_loader, self.checkpoint, [self.device])

                label = ["Pull Up", "Barbell Bench Press", "Barbell Bent-Over Row", "Barbell Upright Row",
                         "Barbell Deadlift", "Barbell Glute Bridge", "Barbell Stiff-leg Deadlift", "Barbell Back Squat",
                         "Barbell Front Squat", "Incline Barbell Bench Press", "Dcline Barbell Bench Press",
                         "Standing Barbell Curl", "Dumbbell Lateral Raise", "Laying Dumbbell Triceps Extension",
                         "Overhead Dumbbell Triceps Extension", "Alternating Dumbbell Curl", "Dumbbell fly",
                         "Seat Dumbbell Shoulder Press ", "Dumbbell Bent-over Row",
                         "Dumbbell Seated Bent-Over Lateral Raise", "Parallel Bars Leg Raise", "Seated Cable Row",
                         "Lat Pull-down", "Cable Front Raise", "Smith-machine Squat", "Smith Machine Shoulder Press ",
                         "Push-Up", "Kneeling Push-Up", "Release Push-Up", "Diamond Push-Up", "Burpee",
                         "Mountain Climbers", "Butt Kick", "Jumping Jack", "Jump Squat", "Standing Oblique Crunch",
                         "Standing Side Crunch", "Single Leg Stretch", "Double Leg Stretch", "Scissor Kick",
                         "Bent Knee Leg Raise", "Side kick（侧踢）", "Plank（平板支撑）", "Criss Cross/Bicycle crunch",
                         "Squat", "Crul-up/Crunch", "Sicilian Crunch（西西里卷腹）", "Reverse curls（反向卷腹）",
                         "Side Leg Lifts（侧卧举腿卷腹）", "Side crunch/Oblique crunch（侧身卷腹）",
                         " Simultaneous knee and abdominal raise/Keen lift crunch（仰卧屈膝两头起）",
                         "Straight leg toe touches（直腿触足卷腹）", "Elbow-to-knee crunch/Cross crunch（仰卧对角交替收膝）",
                         "Russian twist（俄罗斯转体）", "Glute bridge（臀桥）", "Single-Leg glute bridge（单腿臀桥）",
                         "Neck stretch（颈部拉伸）", "Stride stretch（跨步伸展 ）", "Abdominal stretch（腹部拉伸）",
                         "Calf stretch（小腿伸展）", "Lancelot stretch（举臂弓步伸展 ）", "Scarf stretch（肩胛伸展）",
                         "Chest stretch（胸部拉伸）", "Triceps stretch（手臂后侧拉伸）", "Quadriceps stretch（大腿前侧拉伸）",
                         "Hamstring stretch（坐姿开腿伸展/腿筋伸展）", "Side stretch（腹部侧拉）", "Single-Leg hopping（勾腿跳）",
                         "High knees（高抬腿）", "Leg swings（腿部摆动）", "Donkey kicks（驴踢）", "Half Moon Pose（Ardha Chandrasana）",
                         " Downward Dog (Adho Mukha Svanasana) ", "Handstand (Adho Mukha Vrksasana）", "Side Plank Pose（Vasisthasana）",
                         "Child’s Pose（Balasana）", "Garland Pose（Malasana）", "Dance Pose（Natarajasana）",
                         "Standing Wide-Legged Forward Fold Pose（Prasarita Padottanasana）", "Shoulderstand（Salamba sarvangasana）",
                         " Scorpion Pose（Vrschikasanae）", "Swaying Palm Tree Pose（Tiryaka Tadasana）", "Triangle Pose（Trikonasana）",
                         "Chair Pose（Utkatasana ）", "Standing Forward Bend Pose（Uttanasana）", "Warrior I（Virabhadrasana I）",
                         "Warrior II（Virabhadrasana II）", "Warrior III（Virabhadrasana III）"]
                max_k_preds = np.argsort(outputs, axis=1)[:, -1:][:, ::-1]
                pred_label = max_k_preds[0][0]
                print("=============", pred_label, action_label)
                if action_label is not None:
                    if action_label == pred_label:
                        final_msg = 'Action {} of Video {} is standard.'.format(label[action_label], pose_path)
                    elif action_label == pred_label - 88:
                        final_msg = 'Action {} in the video {} is non-standard.'.format(label[action_label], pose_path)
                    else:
                        final_msg = 'It is difficult to assess the Action quality in the video {}.'.format(pose_path)
                    return final_msg
                else:
                    final_msg = 'It is difficult to assess the Action quality in the video {}.'.format(pose_path)
                    return final_msg
            else:
                final_msg = 'Pose {} does not exists.'.format(pose_path)
            return final_msg
        except:
            return "Unkown Error in ActionQuality."

class FSDetect:
    def __init__(self, device):
        self.out = './tools/fire-smoke-detection/result/'
        self.weights = './tools/fire-smoke-detection/best.pt'
        self.save_img = True
        self.imgsz = 640
        self.conf_thres = 0.5
        self.iou_thres = 0.5
        self.ratio = 0.2
        self.device = torch.device(device)
        self.model = attempt_load(self.weights, map_location=self.device)

    @prompts(name="Fire and Smoke Detection",
             description="useful when you want to know whether there is fire or smoke in the video, receives video_path as input. "
                         "The input to this tool should be a string, representing the video_path.")
    def inference(self, video_path):
        try:
            if '/' in video_path:
                source = video_path.split('/')[0] + '/FSDetect_' + video_path.split('/')[-1]
            else:
                final_msg = 'Video {} does not exists.'.format(video_path)
                return final_msg
            # set Dataloader
            vid_path, vid_writer = None, None
            try:
                dataset = LoadImages(source, img_size=self.imgsz)
            except Exception as e:
                msg = e
                return msg, None
            names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

            with torch.no_grad():
                img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
                _ = self.model(img)  # run once
                cla = [False, False]  # mark of fire and smoke
                queue_smoke = queue.Queue(100)
                queue_fire = queue.Queue(100)
                for _ in range(99):
                    queue_smoke.put(0)
                    queue_fire.put(0)
                smoke = 0
                fire = 0
                nframes = 0

                for path, img, im0s, vid_cap in dataset:
                    nframes += 1
                    img = torch.from_numpy(img).to(self.device)
                    img = img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)
                    # Inference
                    pred = self.model(img)[0]
                    # Apply NMS
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        p, s, im0 = path, '', im0s
                        save_path = str(Path(self.out) / Path(p).name)
                        txt_path = str(Path(self.out) / Path(p).stem) + (
                            '_%g' % dataset.frame if dataset.mode == 'video' else '')
                        s += '%gx%g ' % img.shape[2:]  # print string
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        input_fire = False
                        input_smoke = False
                        if det is not None and len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                            # Print results
                            for c in det[:, -1].unique():
                                if names[int(c)] == 'fire':
                                    input_fire = True
                                    fire += 1
                                elif names[int(c)] == 'smoke':
                                    input_smoke = True
                                    smoke += 1
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += '%g %ss, ' % (n, names[int(c)])  # add to string
                            # Write results
                            for *xyxy, conf, cls in det:
                                if self.save_img or self.view_img:  # Add bbox to image
                                    label = '%s %.2f' % (names[int(cls)], conf)
                                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        queue_fire.put(1) if input_fire else queue_fire.put(0)
                        queue_smoke.put(1) if input_smoke else queue_smoke.put(0)
                        if fire / 100 > self.ratio:
                            cla[0] = True
                        if smoke / 100 > self.ratio:
                            cla[1] = True
                        output_fire = queue_fire.get()
                        output_smoke = queue_smoke.get()
                        if output_fire == 1:
                            fire -= 1
                        if output_smoke == 1:
                            smoke -= 1
                        if self.save_img:
                            if dataset.mode == 'images':
                                cv2.imwrite(save_path, im0)
                            else:
                                if vid_path != save_path:  # new video
                                    vid_path = save_path
                                    if isinstance(vid_writer, cv2.VideoWriter):
                                        vid_writer.release()  # release previous video writer

                                    fourcc = 'mp4v'  # output video codec
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                                vid_writer.write(im0)
                if nframes < 100:
                    cla[0] = True if fire / nframes > self.ratio else False
                    cla[1] = True if smoke / nframes > self.ratio else False
                if cla[0] and cla[1]:
                    msg = 'Dangerous! We have detected both smoke and file in video {}.'.format(video_path)
                elif cla[0] and not cla[1]:
                    msg = 'Dangerous! We have detected file in video {}.'.format(video_path)
                elif not cla[0] and cla[1]:
                    msg = 'Dangerous! We have detected smoke in video {}.'.format(video_path)
                else:
                    msg = 'There is no smoke or file detected in video {}.'.format(video_path)

            return msg
        except:
            return "Unkown Error in FSDetect."

class CrowdCounting:
    def __init__(self, device):
        #parser2 = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser_Crowd()])
        self.cfg = Crowdcfg()
        self.output_dir = './results/'
        self.weight_path= './tools/CrowdCounting-P2PNet-main/weights/SHTechA.pth'
        self.device = torch.device(device)
        # get the P2PNet
        self.model = build_model(self.cfg)
        # move to GPU
        self.model.to(self.device)
        # load trained model
        if self.weight_path is not None:
            checkpoint = torch.load(self.weight_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

    @prompts(name="Count People in the Video",
             description="useful when you want to counting the total crowd present in the video, receives video_path as input. "
                         "The input to this tool should be a string, representing the video_path.")
    def inference(self, video_path):
        try:
            pred_count = []
            # create the pre-processing transform
            transform = standard_transforms.Compose([
                standard_transforms.ToTensor(),
                standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            if '/' in video_path:
                video_path = video_path.split('/')[0] + '/FSDetect_' + video_path.split('/')[-1]
            else:
                final_msg = 'Video {} does not exists.'.format(video_path)
                return final_msg
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(self.output_dir, video_name)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                final_msg = 'Video {} does not exist.'.format(video_path)
                return final_msg
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps)
            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    if count % frame_interval == 0:
                        line = []
                        line2 = []
                        image_name = os.path.join(output_path, f"{video_name}_{count}.jpg")
                        cv2.imwrite(image_name, frame)
                        # set your image path here
                        img_path = image_name
                        if os.path.exists(img_path) == False:
                            print("The image doesn't exist!")
                            return False
                        # load the images
                        img_raw = Image.open(img_path).convert('RGB')
                        # round the size
                        width, height = img_raw.size
                        new_width = width // 128 * 128
                        new_height = height // 128 * 128
                        img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
                        # pre-proccessing
                        img = transform(img_raw)

                        samples = torch.Tensor(img).unsqueeze(0)
                        samples = samples.to(self.device)
                        # run inference
                        outputs = self.model(samples)
                        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

                        outputs_points = outputs['pred_points'][0]

                        threshold = 0.5
                        # filter the predictions
                        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
                        predict_cnt = int((outputs_scores > threshold).sum())

                        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

                        outputs_points = outputs['pred_points'][0]
                        size = 5
                        img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
                        for p in points:
                            img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
                        cv2.imwrite(os.path.join(output_path, f"{video_name}_{count}.jpg"), img_to_draw)
                        pred_count.append(predict_cnt)
                    count += 1
                else:
                    break
            cap.release()
            final_msg = "There are about {} to {} people in the video.".format(np.min(pred_count), np.max(pred_count))
            return final_msg
        except:
            return "Unkown Error in Crowd Counting."
