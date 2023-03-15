import os 

import numpy as np

import torch
import pytorch_lightning as pl
import onnx
import onnxruntime

from models import resnet18
from module.classifier import Classifier
from utils.yaml_helper import get_train_configs

import platform
import argparse

def export_onnx(cfg, args):
    input_size = cfg['input_size']
    
    dst_fname = args.savedmodel.replace('.ckpt', '.onnx')
    
    model = resnet18.ResNet_FC(resnet_version='resnet18', output_channel=cfg['classes'])
    model_module = Classifier.load_from_checkpoint(
        args.savedmodel, model=model
    )
    model_module.eval() # or model_module train(False)
    
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    # Referenced novaIC Tool Guide documents
    torch.onnx.export(
        model, dummy_input, dst_fname,
        opset_version=12, verbose=False, input_names=["input_1"],
        export_params=True, do_constant_folding=False
    )
    
    onnx_model = onnx.load(dst_fname)
    onnx.checker.check_model(onnx_model)
    
    onnx_session = onnxruntime.InferenceSession(dst_fname)
    output = onnx_session.run(None, {'input_1': dummy_input.numpy()})  # optional
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='export_onnx')
    parser.add_argument('--cfg', default='./config/resnet18_cifar10.yaml', type=str)
    parser.add_argument('--savedmodel', type=str)
    args = parser.parse_args()
    
    cfg = get_train_configs(args.cfg)
    
    export_onnx(cfg, args)