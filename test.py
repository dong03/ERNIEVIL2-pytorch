#!/usr/bin/env python
# encoding: utf-8
"""
File Description:
Verify consistency before and after converting
official repo: https://github.com/PaddlePaddle/ERNIE/tree/ernie-kit-open-v1.0/Research/ERNIE-ViL2
Author: chengbo dong
Mail: cb_dong@foxmail.com
Created Time: 2023/1/6
"""
import sys
import yaml
from PIL import Image
from attrdict import AttrDict

import paddle
import paddlenlp
import torch
import torch.nn as nn

import transformers
from transformers import ErnieConfig, ViTImageProcessor
from ernievil2torch.transformers.ERNIE import ErnieModel
from ernievil2.transformers.multimodal import ERNIE_ViL2_base
from ernievil2torch.transformers.ViT import ViT_base_patch16_224

    
class ERNIEVIL2(nn.Module):
    def __init__(self):
        super().__init__()
        config = ErnieConfig.from_json_file('/home/dcb/code/bv/ERNIE-ViL2/packages/ernie_base_3.0/ernie_config.base.json')
        self.visual = ViT_base_patch16_224()
        self.ernie = ErnieModel(config = config)
    def forward(self, x):
        return x

def performance_check():
    torch_model = ERNIEVIL2()
    ckpt = torch.load('ERNIE_VIL2_BASE_ViT_convert.pt', map_location='cpu')
    torch_model.load_state_dict(ckpt)
    torch_model.eval()
    image_processor = ViTImageProcessor(image_mean=[0.485, 0.456, 0.406],
                  image_std= [0.229, 0.224, 0.225])
    tokenizer = transformers.BertTokenizer.from_pretrained('nghuyong/ernie-3.0-base-zh') 
    
    image_path = ['src/cat.jpg','src/dog.jpg', 'src/cyberpunk.jpg']
    images = [Image.open(each).convert('RGB') for each in image_path]
    image_input = image_processor(images=images, return_tensors="pt")
    
    texts = ['一张猫的照片', '一张狗的照片',"赛博朋克边缘行者"]    
    inputs = tokenizer(text=texts, padding=True, return_tensors="pt")
    
    with torch.no_grad():
        visual_output = torch_model.visual(image_input['pixel_values'])[:, 0]
        title_features = torch_model.ernie(**inputs).pooler_output
    
    title_features = title_features / title_features.norm(p=2, dim=-1, keepdim=True)  # normalize
    visual_output = visual_output / visual_output.norm(p=2, dim=-1, keepdim=True)  # normalize
    print(title_features @ visual_output.T)

def logit_check(args):
    torch_model = ERNIEVIL2()
    ckpt = torch.load('ERNIE_VIL2_BASE_ViT_convert.pt', map_location='cpu')
    torch_model.load_state_dict(ckpt)
    torch_model.eval()
    ## torch_visual
    image_processor = ViTImageProcessor(image_mean=[0.485, 0.456, 0.406],
                  image_std= [0.229, 0.224, 0.225])
    
    image = Image.open('src/dog.jpg').convert('RGB')
    image_input = image_processor(images=[image], return_tensors="pt")
    with torch.no_grad():
        t_visual_output = torch_model.visual(image_input['pixel_values'])
    
    ## torch_text
    tokenizer = transformers.BertTokenizer.from_pretrained('nghuyong/ernie-3.0-base-zh')

    input_ids = torch.tensor([tokenizer.encode(text="welcome to ernie pytorch project", add_special_tokens=True)])
    with torch.no_grad():
        t_text_output = torch_model.ernie(input_ids)
    print("### pytorch result")
    print("visual_output: ", t_visual_output[:,0][:,:10].numpy())
    print("text_output: ", t_text_output.pooler_output[:,:10].numpy())
    
    ## paddle_visual
    place = "cpu"
    paddle.set_device(place)
    paddle_model = ERNIE_ViL2_base(args)
    paddle_model.eval()
    with paddle.no_grad():
        p_visual_output = paddle_model.visual(paddle.to_tensor(image_input['pixel_values'].numpy()))

    tokenizer = paddlenlp.transformers.AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
    inputs = tokenizer("welcome to ernie pytorch project")
    inputs = {k: paddle.to_tensor([v]) for (k, v) in inputs.items()}
    
    with paddle.no_grad():
        p_text_output = paddle_model.text_model(inputs['input_ids'])
    import pdb; pdb.set_trace()
    print("### paddle result")
    print("visual_output: ", p_visual_output[:,0][:,:10].numpy())
    print("text_output: ", p_text_output[0][:,:10].numpy())

    
if __name__ == '__main__':
    import sys
    with open('./packages/configs/ernie_vil_base.yaml', 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
    if sys.argv[1] == 'logit_check':
        logit_check(args)
    elif sys.argv[1] == 'performance_check':
        performance_check()