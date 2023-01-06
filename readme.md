<h1 align="center">ERNIEVIL2-Pytorch</h1>

<p align="center">This project is to convert ERNIE-VIL2 from paddlepaddle to pytorch format.</p>

<p align="center">
  <a href="https://github.com/dong03/ERNIEVIL2-pytorch/stargazers">
    <img src="https://img.shields.io/github/stars/dong03/ERNIEVIL2-pytorch.svg?colorA=orange&colorB=orange&logo=github"
         alt="GitHub stars">
  </a>
  <a href="https://github.com/dong03/ERNIEVIL2-pytorch/issues">
        <img src="https://img.shields.io/github/issues/dong03/ERNIEVIL2-pytorch.svg"
             alt="GitHub issues">
  </a>
  <a href="https://github.com/dong03/ERNIEVIL2-pytorch/">
        <img src="https://img.shields.io/github/last-commit/dong03/ERNIEVIL2-pytorch.svg">
  </a>
   <a href="https://github.com/dong03/ERNIEVIL2-pytorch/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/dong03/ERNIEVIL2-pytorch.svg">
  </a>
  
</p>



## Requirements
- ```pip install transformers, torch```
- Follow requirements in [official repo](https://github.com/PaddlePaddle/ERNIE/tree/ernie-kit-open-v1.0/Research/ERNIE-ViL2).
- download paddle & torch ckpt from [BaiduYun](链接：https://pan.baidu.com/s/1fqt51Gisra6Rqk9OCC0ovQ?pwd=kaxr).

## Evaluate
**To conduct cross-modal similarity computation,** 
```
python test.py performance_check
```
You will get a similarity matrix between three images and three sentences, like this,
```
tensor([[0.3096, 0.1929, 0.1588],
        [0.2270, 0.2997, 0.1339],
        [0.0894, 0.1035, 0.3198]])
```
**To check the calculation results before and after model conversion,**

```bash
python check.py logit_check
```

You will get the output:

```output
### pytorch result
visual_output:  [[-0.25474143 -0.72782516  0.02674462  0.48610407  1.4485253   0.5175752
   1.0823581   0.3140268   0.32782146  0.4190097 ]]
text_output:  [[ 0.10204837 -0.5075943  -0.05125085  0.22701152  0.5774069  -0.54781747
  -0.1122973   0.46482086  0.2952882   0.1963322 ]]

### paddle result
visual_output: [[-0.25474280, -0.72782505,  0.02674395,  0.48610494,  1.44852710, 0.51757193,  1.08235860,  0.31402659,  0.32782283,  0.41901165]]
text_output: [[ 0.10174122, -0.50740963, -0.05135643,  0.22727829,  0.57717800, -0.54775286, -0.11231337,  0.46472329,  0.29527009,  0.19644395]]

```

It can be seen that the result of our convert version is the same with the official paddlepaddle's version.

## Change Describe

### **Text encoder**:
This part mainly follows [ERNIE-Pytorch](https://github.com/nghuyong/ERNIE-Pytorch)'s pipeline. It has also been merged into [huggingface/transformers@v4.22.0](https://github.com/huggingface/transformers/releases/tag/v4.22.0).

However, key names in model state_dict and [ErnieEmbeddings](ernievil2torch/transformers/ERNIE.py) are slightly changed to suit paddle version's ERNIE-VIL2.

### **Visual Encoder**
I rewrite the paddle version of VIT as it's quite difference wth huggingface's. Some initialization functions are discarded as normally we'll start from the pretrained checkpoint.(And I have no idea of their pytorch version)

### **Do it yourself**
If you want to convert model on your own, 

1. download paddle version [checkpoint](http://bj.bcebos.com/wenxin-models/ERNIE_VIL2_BASE_ViT.pdparams)
2. ```python convert.py```


## Citation

If you use this work in a scientific publication, I would appreciate that you can also cite the following BibTex entries:

```latex
@misc{dong03@ERNIEVIL2-Pytorch,
  title={ERNIEVIL2-Pytorch},
  author={Chengbo Dong},
  howpublished={\url{https://github.com/nghuyong/ERNIE-Pytorch}},
  year={2023}
}
```

(or at least 
```\footnote{\url{http}}``` 😆