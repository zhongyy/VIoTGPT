CUDA_VISIBLE_DEVICES='0' python test_VIoTGPT_upload.py \
--load FaceRecognition_cuda:0,PersonReid_cuda:0,GaitRecognition_cuda:0,VehicleReid_cuda:0,PlateRecognition_cuda:0,CrowdCounting_cuda:0,FSDetect_cuda:0,SceneRecognition_cuda:0,AnomalyDetection_cuda:0,HumanPose_cuda:0,HumanAction_cuda:0 \
--query_path ./VIoT_tool/test_face.json \
--query_data_path ./VIoT_tool/Data/FR/ \
--model_path ./models/vicuna_7B/ \
--lora_path ./train/output/train_vicuna-7b_lora/ \
--output_path ./evaluation/fr/train_vicuna-7b_lora.json \
2>&1|tee ./evaluation/fr/train_vicuna-7b_lora.log

CUDA_VISIBLE_DEVICES='0' python test_VIoTGPT_upload.py \
--load FaceRecognition_cuda:0,PersonReid_cuda:0,GaitRecognition_cuda:0,VehicleReid_cuda:0,PlateRecognition_cuda:0,CrowdCounting_cuda:0,FSDetect_cuda:0,SceneRecognition_cuda:0,AnomalyDetection_cuda:0,HumanPose_cuda:0,HumanAction_cuda:0 \
--query_path ./VIoT_tool/test_personreid.json \
--query_data_path ./VIoT_tool/Data/person/ \
--model_path ./models/vicuna_7B/ \
--lora_path ./train/output/train_vicuna-7b_lora/ \
--output_path ./evaluation/person/train_vicuna-7b_lora.json \
2>&1|tee ./evaluation/person/train_vicuna-7b_lora.log

CUDA_VISIBLE_DEVICES='0' python test_VIoTGPT_uploadvideo.py \
--load FaceRecognition_cuda:0,PersonReid_cuda:0,GaitRecognition_cuda:0,VehicleReid_cuda:0,PlateRecognition_cuda:0,CrowdCounting_cuda:0,FSDetect_cuda:0,SceneRecognition_cuda:0,AnomalyDetection_cuda:0,HumanPose_cuda:0,HumanAction_cuda:0 \
--query_path ./VIoT_tool/test_gait.json \
--model_path ./models/vicuna_7B/ \
--lora_path ./train/output/train_vicuna-7b_lora/ \
--output_path ./evaluation/gait/train_vicuna-7b_lora.json \
2>&1|tee ./evaluation/gait/train_vicuna-7b_lora.log

CUDA_VISIBLE_DEVICES='0' python test_VIoTGPT_noupload.py \
--load FaceRecognition_cuda:0,PersonReid_cuda:0,GaitRecognition_cuda:0,VehicleReid_cuda:0,PlateRecognition_cuda:0,CrowdCounting_cuda:0,FSDetect_cuda:0,SceneRecognition_cuda:0,AnomalyDetection_cuda:0,HumanPose_cuda:0,HumanAction_cuda:0 \
--query_path ./VIoT_tool/test_licenceplate.json \
--model_path ./models/vicuna_7B/ \
--lora_path ./train/output/train_vicuna-7b_lora/ \
--output_path ./evaluation/licenceplate/train_vicuna-7b_lora.json \
2>&1|tee ./evaluation/licenceplate/train_vicuna-7b_lora.log

CUDA_VISIBLE_DEVICES='0' python test_VIoTGPT_upload.py \
--load FaceRecognition_cuda:0,PersonReid_cuda:0,GaitRecognition_cuda:0,VehicleReid_cuda:0,PlateRecognition_cuda:0,CrowdCounting_cuda:0,FSDetect_cuda:0,SceneRecognition_cuda:0,AnomalyDetection_cuda:0,HumanPose_cuda:0,HumanAction_cuda:0 \
--query_path ./VIoT_tool/test_vehiclereid.json \
--query_data_path ./Data/vehicle/ \
--model_path ./models/vicuna_7B/ \
--lora_path ./train/output/train_vicuna-7b_lora/ \
--output_path ./evaluation/vehiclereid/train_vicuna-7b_lora.json \
2>&1|tee ./evaluation/vehiclereid/train_vicuna-7b_lora.log

CUDA_VISIBLE_DEVICES='0' python test_VIoTGPT_noupload.py \
--load FaceRecognition_cuda:0,PersonReid_cuda:0,GaitRecognition_cuda:0,VehicleReid_cuda:0,PlateRecognition_cuda:0,CrowdCounting_cuda:0,FSDetect_cuda:0,SceneRecognition_cuda:0,AnomalyDetection_cuda:0,HumanPose_cuda:0,HumanAction_cuda:0 \
--query_path ./VIoT_tool/test_crowd.json \
--model_path ./models/vicuna_7B/ \
--lora_path ./train/output/train_vicuna-7b_lora/ \
--output_path ./evaluation/crowd/train_vicuna-7b_lora.json \
2>&1|tee ./evaluation/crowd/train_vicuna-7b_lora.log

CUDA_VISIBLE_DEVICES='0' python test_VIoTGPT_noupload.py \
--load FaceRecognition_cuda:0,PersonReid_cuda:0,GaitRecognition_cuda:0,VehicleReid_cuda:0,PlateRecognition_cuda:0,CrowdCounting_cuda:0,FSDetect_cuda:0,SceneRecognition_cuda:0,AnomalyDetection_cuda:0,HumanPose_cuda:0,HumanAction_cuda:0 \
--query_path ./VIoT_tool/test_fire.json \
--model_path ./models/vicuna_7B/ \
--lora_path ./train/output/train_vicuna-7b_lora/ \
--output_path ./evaluation/fire/train_vicuna-7b_lora.json \
2>&1|tee ./evaluation/fire/train_vicuna-7b_lora.log

CUDA_VISIBLE_DEVICES='0' python test_VIoTGPT_noupload.py \
--load FaceRecognition_cuda:0,PersonReid_cuda:0,GaitRecognition_cuda:0,VehicleReid_cuda:0,PlateRecognition_cuda:0,CrowdCounting_cuda:0,FSDetect_cuda:0,SceneRecognition_cuda:0,AnomalyDetection_cuda:0,HumanPose_cuda:0,HumanAction_cuda:0,ActionQuality_cuda:0 \
--query_path ./VIoT_tool/test_anomaly.json \
--model_path ./models/vicuna_7B/ \
--lora_path ./train/output/train_vicuna-7b_lora/ \
--output_path ./evaluation/anomaly/train_vicuna-7b_lora.json \
2>&1|tee ./evaluation/anomaly/train_vicuna-7b_lora.log

CUDA_VISIBLE_DEVICES='0' python test_VIoTGPT_noupload.py \
--load FaceRecognition_cuda:0,PersonReid_cuda:0,GaitRecognition_cuda:0,VehicleReid_cuda:0,PlateRecognition_cuda:0,CrowdCounting_cuda:0,FSDetect_cuda:0,SceneRecognition_cuda:0,AnomalyDetection_cuda:0,HumanPose_cuda:0,HumanAction_cuda:0,ActionQuality_cuda:0 \
--query_path ./VIoT_tool/test_action.json \
--model_path ./models/vicuna_7B/ \
--lora_path ./train/output/train_vicuna-7b_lora/ \
--output_path ./evaluation/action/train_vicuna-7b_lora.json \
2>&1|tee ./evaluation/action/train_vicuna-7b_lora.log





