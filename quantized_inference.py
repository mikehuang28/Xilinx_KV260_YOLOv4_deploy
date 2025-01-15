# quantized model inference
import torch
import torch.nn as nn
import numpy as np
import cv2
import importlib
import random
import argparse
from quantized_utils import (
    YOLOPost,
    non_max_suppression,
    # non_max_suppression_with_class_thresholds,
)

import pytorch_nndct

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################################################
# load parser information
# pt file size and nms thres
parser = argparse.ArgumentParser()
parser.add_argument("--pt_file_size", type=int, default=224, help="pt file size")
parser.add_argument(
    "--nms_threshold", type=float, default=0.01, help="IOU threshold for NMS"
)
parser.add_argument("--pt_file_type", type=str, default="best", help="pt file type")
opt = parser.parse_args()
##################################################

# Load the model
# model = torch.jit.load(
#     "./quantize_result/Darknet_int.pt", map_location=torch.device("cpu")
# )

pt_file_prefix = "Darknet_int_"
pt_type = opt.pt_file_type
pt_file_size = opt.pt_file_size
pt_file_suffix = ".pt"
pt_file = pt_file_prefix + str(pt_type) + "_" + str(pt_file_size) + pt_file_suffix
suffix = pt_file.split("int")[1].split(".pt")[0]
print(pt_file)
# print(pt_file.split("int_")[1].split(".pt")[0])
model = torch.jit.load(
    "./models/best_quantized_pt/" + pt_file, map_location=torch.device("cpu")
)
# print(model)
model.eval()
model = model.to(device)


####### Load the config
params_path = "params.py"
config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS


##################################################
# conf
conf_threshold = config["confidence_threshold"]
nms_threshold = opt.nms_threshold
##################################################


# YOLO loss with 3 scales
yolo_losses = []
for i in range(3):
    # yolo_losses.append(YOLOLoss(config["yolo"]["anchors"][i],
    #                             config["yolo"]["classes"], (config["img_w"], config["img_h"])))

    yolo_losses.append(
        YOLOPost(
            config["yolo"]["anchors"][i],
            config["yolo"]["classes"],
            (config["img_w"], config["img_h"]),
        )
    )

# print(yolo_losses)

####### Inference
# Pre-processing
##################################################
# enter image name
image_name = "VCH01_20220927_102121_894_1551"  # road best 0.2 0.05
# image_name = "VCH01_20220927_085140_483_911"  # sunny
# image_name = "VCH01_20220913_105449_585_191"  # cloudy best 0.25 0.05
# image_name = "VCH01_20220913_184323_384_1801"  # night

# VCH01_20220927_102121_894_1571
##################################################

# freeway
# image_name = "20221013143518_r_331"  # best 0.4 0.1

# load image
image_path = "./images/images_alan/" + image_name + ".jpg"
# image_path = "./images/images_freeway/" + image_name + ".jpg"
print(image_path)
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(
    image, (config["img_w"], config["img_h"]), interpolation=cv2.INTER_LINEAR
)
image = image.astype(np.float32)
image /= 255.0
image = np.transpose(image, (2, 0, 1))
image = image.astype(np.float32)

# print(image.shape) # (3,416,416)

image = torch.from_numpy(image)
image = image.unsqueeze(0)  # ([1,3,416,416])

# print(image.shape)  # ([1,3,416,416])
print(f"image size = {image.shape}")


####### Inference

# Perform inference
with torch.no_grad():
    out = model(image.to(device))

# print(out)

# print(out[0].shape) # ([1,10647,11]) -> torch.Tensor
# print(out[1].shape) # ([1,3,52,52,11]) -> torch.Tensor
# print(out[2].shape) # ([1,3,26,26,11]) -> torch.Tensor
# print(out[3].shape) # ([1,3,13,13,11]) -> torch.Tensor

# print(out[0].shape)  # ([1,33,28,28]) -> torch.Tensor
# print(out[1].shape)  # ([1,33,14,14]) -> torch.Tensor
# print(out[2].shape)  # ([1,33,7,7]) -> torch.Tensor

# Convert tensor to numpy
output = []
output.append(out[0].numpy())
output.append(out[1].numpy())
output.append(out[2].numpy())
# print(len(output))
# print(output[0].shape)  # (1, 255, 13, 13) -> numpy

output_list = []
for i in range(3):
    output_list.append(yolo_losses[i].forward(output[i]))
output_con = np.concatenate(output_list, 1)  # concat at axis 1


# # print(len(output_list))
# print(output_list[0].shape)  # ([1, 507, 85]) | 13*13*3
# print(output_list[1].shape)  # ([1, 2028, 85]) | 26*26*3
# print(output_list[2].shape)  # ([1, 8112, 85]) | 52*52*3

# # print(len(output_con))
# # print(output_con.shape) # ([1, 10647, 85])

# output_con = out[0].numpy()  # Convert tensor to numpy
print(output_con.shape)

# nms
batch_detections = non_max_suppression(
    output_con,  # prediction
    config["yolo"]["classes"],  # classes
    conf_thres=config["confidence_threshold"],  # confidence threshold
    nms_thres=nms_threshold,  # iou threshold
)


print("-----------------")
print(f"conf_threshold = {conf_threshold}, nms_threshold = {nms_threshold}")

# print(f"batch_detections = {batch_detections}")
# print(batch_detections)

######## Plot prediction with bounding box
classes = open(config["classes_names_path"], "r").read().split("\n")[:-1]

# bbox colors for each class
# BGR
bbox_class_colors = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (0, 255, 255),
    5: (255, 0, 255),
}

for idx, detections in enumerate(batch_detections):
    if detections is not None:
        im = cv2.imread(image_path)
        # print(im.shape)  # eg. (428, 640, 3)
        unique_labels = np.unique(detections[:, -1])
        # print(detections[:, -1])
        n_cls_preds = len(unique_labels)
        print(unique_labels)
        for i in unique_labels:
            print(classes[int(i)])  # print class name

        # bbox_colors = {
        #     int(cls_pred): (
        #         random.randint(0, 255),
        #         random.randint(0, 255),
        #         random.randint(0, 255),
        #     )
        #     for cls_pred in unique_labels
        # }

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            # color = bbox_colors[int(cls_pred)]
            color = bbox_class_colors[int(cls_pred)]  # fixed bbox colors

            # Rescale coordinates to original dimensions
            ori_h, ori_w, _ = im.shape
            pre_h, pre_w = config["img_h"], config["img_w"]
            box_h = ((y2 - y1) / pre_h) * ori_h
            box_w = ((x2 - x1) / pre_w) * ori_w
            y1 = (y1 / pre_h) * ori_h
            x1 = (x1 / pre_w) * ori_w

            # Create a Rectangle patch
            cv2.rectangle(
                im, (int(x1), int(y1)), (int(x1 + box_w), int(y1 + box_h)), color, 2
            )

            # Add label
            label = classes[int(cls_pred)]
            cv2.putText(
                im,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    # Save generated image with detections
    output_path = (
        "./prediction/"
        + str(pt_type)
        + "/pred_"
        + image_name
        + suffix
        + "_"
        + str(conf_threshold)
        + "_"
        + str(nms_threshold)
        + ".jpg"
    )

    print(f"output path: {output_path}")

    # Add file name on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)  # white
    font_thickness = 1
    x, y = 10, 30  # xy coordinate
    cv2.putText(im, output_path, (x, y), font, font_scale, font_color, font_thickness)

    # save
    cv2.imwrite(output_path, im)


print("--- quantization inference done!")
