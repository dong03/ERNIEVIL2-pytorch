#!/usr/bin/env python
# encoding: utf-8
"""
File Description:
convert ernie-vil2 into pytorch version
official repo: https://github.com/PaddlePaddle/ERNIE/tree/ernie-kit-open-v1.0/Research/ERNIE-ViL2
Author: chengbo dong
Mail: cb_dong@foxmail.com
Created Time: 2023/1/6
"""
import collections
import paddle.fluid.dygraph as D
import torch
from paddle import fluid
import collections

def build_params_map(attention_num=12):
    """
    build params map from paddle-paddle's ERNIE to transformer's BERT
    """
    weight_map = collections.OrderedDict({})
    
    # text_model
    weight_map.update({
        'text_model.word_emb.weight': "ernie.embeddings.word_embeddings.weight",
        'text_model.pos_emb.weight': "ernie.embeddings.position_embeddings.weight",
        'text_model.ln.weight': 'ernie.embeddings.LayerNorm.weight',
        'text_model.ln.bias': 'ernie.embeddings.LayerNorm.bias',
    })
    # add attention layers
    for i in range(attention_num):
        weight_map[f'text_model.encoder_stack.block.{i}.attn.q.weight'] = f'ernie.encoder.layer.{i}.attention.self.query.weight'
        weight_map[f'text_model.encoder_stack.block.{i}.attn.q.bias'] = f'ernie.encoder.layer.{i}.attention.self.query.bias'
        weight_map[f'text_model.encoder_stack.block.{i}.attn.k.weight'] = f'ernie.encoder.layer.{i}.attention.self.key.weight'
        weight_map[f'text_model.encoder_stack.block.{i}.attn.k.bias'] = f'ernie.encoder.layer.{i}.attention.self.key.bias'
        weight_map[f'text_model.encoder_stack.block.{i}.attn.v.weight'] = f'ernie.encoder.layer.{i}.attention.self.value.weight'
        weight_map[f'text_model.encoder_stack.block.{i}.attn.v.bias'] = f'ernie.encoder.layer.{i}.attention.self.value.bias'
        weight_map[f'text_model.encoder_stack.block.{i}.attn.o.weight'] = f'ernie.encoder.layer.{i}.attention.output.dense.weight'
        weight_map[f'text_model.encoder_stack.block.{i}.attn.o.bias'] = f'ernie.encoder.layer.{i}.attention.output.dense.bias'
        
        weight_map[f'text_model.encoder_stack.block.{i}.ln1.weight'] = f'ernie.encoder.layer.{i}.attention.output.LayerNorm.weight'
        weight_map[f'text_model.encoder_stack.block.{i}.ln1.bias'] = f'ernie.encoder.layer.{i}.attention.output.LayerNorm.bias'
        
        weight_map[f'text_model.encoder_stack.block.{i}.ffn.i.weight'] = f'ernie.encoder.layer.{i}.intermediate.dense.weight'
        weight_map[f'text_model.encoder_stack.block.{i}.ffn.i.bias'] = f'ernie.encoder.layer.{i}.intermediate.dense.bias'
        weight_map[f'text_model.encoder_stack.block.{i}.ffn.o.weight'] = f'ernie.encoder.layer.{i}.output.dense.weight'
        weight_map[f'text_model.encoder_stack.block.{i}.ffn.o.bias'] = f'ernie.encoder.layer.{i}.output.dense.bias'
        weight_map[f'text_model.encoder_stack.block.{i}.ln2.weight'] = f'ernie.encoder.layer.{i}.output.LayerNorm.weight'
        weight_map[f'text_model.encoder_stack.block.{i}.ln2.bias'] = f'ernie.encoder.layer.{i}.output.LayerNorm.bias'
    
    # add pooler
    weight_map.update(
        {
            'text_model.pooler.weight': 'ernie.pooler.dense.weight',
            'text_model.pooler.bias': 'ernie.pooler.dense.bias',
        }
    )
    return weight_map



if __name__ == '__main__':
    with fluid.dygraph.guard():
        paddle_paddle_params, _ = D.load_dygraph('ERNIE_VIL2_BASE_ViT.pdparams')
    weight_map = build_params_map()
    state_dict = collections.OrderedDict()
    
    for weight_name, weight_value in paddle_paddle_params.items():
        if 'text_model' in weight_name:
            if 'weight' in weight_name and ('text_model.encoder_stack' in weight_name or 'text_model.pooler' in weight_name):
                weight_value = weight_value.transpose()
            if weight_name not in weight_map:
                print('=' * 20, '[SKIP]', weight_name, '=' * 20)
                continue
            state_dict[weight_map[weight_name]] = torch.FloatTensor(weight_value)
            print(weight_name, '->', weight_map[weight_name], weight_value.shape)
    
        elif 'visual' in weight_name:
            if 'weight' in weight_name and ('blocks' in weight_name or 'pooler' in weight_name):
                weight_value = weight_value.transpose()
            state_dict[weight_name] = torch.FloatTensor(weight_value)
    
    state_dict['ernie.embeddings.position_ids'] = torch.arange(2048).expand((1, -1))
    del state_dict['visual.classfy.weight']
    del state_dict['visual.classfy.bias']
    torch.save(state_dict, "ERNIE_VIL2_BASE_ViT_convert.pt")
    print("finish converting")
    
    
