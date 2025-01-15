# reference:https://github.com/Xilinx/Vitis-AI/blob/master/src/vai_quantizer/vai_q_pytorch/example/resnet18_quant.py
# Do not use this code to quantize yolov4-csp due to the poor performance after quantization and compilation
import os
import re
import sys
import argparse
import time
import pdb
import random
from pytorch_nndct.apis import torch_quantizer, dump_xmodel
import torch
import torchvision
import torchvision.transforms as transforms

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random


from utils.google_utils import attempt_load

from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
)  # non_max_suppression_all,
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

# from models.models import *
from models.models import *
from utils.datasets import *
from utils.general import *
from torch import jit
from tqdm import tqdm
from test import *

# device = torch.device("cuda")
# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_dir",
    default="./images/images_alan/",
    help="Data set directory, when quant_mode=calib, it is for calibration, while quant_mode=test it is for evaluation",
)
parser.add_argument(
    "--model_dir",
    default="models",
    help="Trained model file path. Download pretrained model from the following url and put it in model_dir specified path: https://download.pytorch.org/models/resnet18-5c106cde.pth",
)
parser.add_argument(
    "--subset_len",
    default=None,
    type=int,
    help="subset_len to evaluate model, using the whole validation dataset if it is not set",
)
parser.add_argument(
    "--batch_size", default=32, type=int, help="input data batch size to evaluate model"
)
parser.add_argument(
    "--quant_mode",
    default="calib",
    choices=["float", "calib", "test"],
    help="quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model",
)
parser.add_argument(
    "--fast_finetune",
    dest="fast_finetune",
    action="store_true",
    help="fast finetune model before calibration",
)
parser.add_argument(
    "--deploy", dest="deploy", action="store_true", help="export xmodel for deployment"
)
parser.add_argument(
    "--config_file", default=None, help="quantization configuration file"
)
# add from alan's code
parser.add_argument("--cfg", type=str, help="*.cfg path")
parser.add_argument("--img_size", type=int, default=224, help="inference size (pixels)")
parser.add_argument(
    "--weights",
    nargs="+",
    type=str,
    default="/home/310581001/runs/train/exp126/weights/last.pt",
    help="model.pt path(s)",
)

args, _ = parser.parse_known_args()


def quantization(title="optimize", model_name="", file_path=""):
    data_dir = args.data_dir
    quant_mode = args.quant_mode
    finetune = args.fast_finetune
    deploy = args.deploy
    batch_size = args.batch_size
    subset_len = args.subset_len
    config_file = args.config_file

    # Assertions
    if quant_mode != "test" and deploy:
        deploy = False
        print(
            r"Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!"
        )
    if deploy and (batch_size != 1 or subset_len != 1):
        print(
            r"Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!"
        )
        batch_size = 1
        subset_len = 1

    # model = Yolov4(yolov4conv137weight=None, n_classes=80, inference=True).cpu()
    # model.load_state_dict(torch.load(file_path))
    # model.aux_logits = False

    ####################################################################################

    weights = args.weights
    cfg = args.cfg
    imgsz = args.img_size
    print(f"image size = {imgsz}")
    # Load model
    model = Darknet(cfg, imgsz).cpu()
    # print(model)
    # model.load_state_dict(torch.load(weights[0], map_location=device)["model"])
    # model.load_state_dict(torch.load(file_path))

    # model.to(device).eval()
    try:
        model.load_state_dict(torch.load(weights[0], map_location=device)["model"])
    #     # model = attempt_load(weights, map_location=device)  # load FP32 model
    #     # imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    except:
        load_darknet_weights(model, weights[0])
    model.aux_logits = False
    model.to(device).eval()
    # if half:
    #     model.half()  # to FP16

    example_input = torch.randn([batch_size, 3, imgsz, imgsz])

    if quant_mode == "float":
        quant_model = model
    else:
        ## new api
        ####################################################################################
        quantizer = torch_quantizer(
            quant_mode,
            model,
            (example_input),
            device=device,
            custom_quant_ops=[
                "aten::softplus",
                # "aten::pow",
                # "aten::mul_",
                # "nndct_strided_slice_inplace_copy",
            ],
            quant_config_file=config_file,
        )
        # print("finish quantizer!")  # add
        quant_model = quantizer.quant_model
        #####################################################################################

    # to get loss value after evaluation
    # loss_fn = torch.nn.CrossEntropyLoss().to(device)
    # print(loss_fn)
    # print(type(loss_fn))

    # val_loader, _ = load_data(
    #     subset_len=subset_len,
    #     train=False,
    #     batch_size=batch_size,
    #     sample_method="random",
    #     data_dir=data_dir,
    #     model_name=model_name,
    # )

    # fast finetune model or load finetuned parameter before test
    # if finetune == True:
    #     ft_loader, _ = load_data(
    #         subset_len=1024,
    #         train=False,
    #         batch_size=batch_size,
    #         sample_method=None,
    #         data_dir=data_dir,
    #         model_name=model_name,
    #     )
    #     if quant_mode == "calib":
    #         #   quantizer.fast_finetune(evaluate, (quant_model, ft_loader, loss_fn))
    #         quantizer.fast_finetune(evaluate, (quant_model, ft_loader))
    #     elif quant_mode == "test":
    #         quantizer.load_ft_param()

    # record  modules float model accuracy
    # add modules float model accuracy here
    # acc_org1 = 0.0
    # acc_org5 = 0.0
    # loss_org = 0.0

    # register_modification_hooks(model_gen, train=False)
    # acc1_gen, acc5_gen, loss_gen = evaluate(quant_model, val_loader, loss_fn)
    # acc1_gen, acc5_gen = evaluate(quant_model, val_loader)

    # logging accuracy
    # print('loss: %g' % (loss_gen))
    # print("top-1 / top-5 accuracy: %g / %g" % (acc1_gen, acc5_gen))

    # Forward -- Dry Run
    # input_data = torch.randn([batch_size, 3, 416, 416]).to(device)
    input_data = torch.randn([batch_size, 3, imgsz, imgsz]).to(device)
    quant_model(input_data)

    # handle quantization result
    if quant_mode == "calib":
        quantizer.export_quant_config()
    if deploy:
        quantizer.export_torch_script()
        quantizer.export_onnx_model()
        quantizer.export_xmodel(deploy_check=False)


if __name__ == "__main__":
    model_name = "best"
    # model_name = "yolov4"
    # file_path = os.path.join(args.model_dir, model_name + ".pt")
    file_path = "models/best.pt"

    feature_test = " float model evaluation"
    if args.quant_mode != "float":
        feature_test = " quantization"
        # force to merge BN with CONV for better quantization accuracy
        args.optimize = 1
        feature_test += " with optimization"
    else:
        feature_test = " float model evaluation"
    title = model_name + feature_test

    print("-------- Start {} test ".format(model_name))

    # calibration or evaluation
    quantization(title=title, model_name=model_name, file_path=file_path)

    print("-------- End of {} test ".format(model_name))
