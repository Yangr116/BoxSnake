# BoxSnake: Polygonal Instance Segmentation with Box Supervision
[Rui Yang](https://yangr116.github.io), [Lin Song](http://linsong.info), [Yixiao Ge](https://geyixiao.com), [Xiu Li](https://www.sigs.tsinghua.edu.cn/lx/main.htm)

**BoxSnake** is an end-to-end training technique to achieve effective **polygonal instance segmentation using only box annotations**. It consists of two loss functions: (1) a point-based unary loss that constrains the bounding box of predicted polygons to achieve coarse-grained segmentation; and (2) a distance-aware pairwise loss that encourages the predicted polygons to fit the object boundaries.

![Intro](assets/BoxSnake.png)
[Arxiv Paper](https://arxiv.org/pdf/2303.11630.pdf) | [Video Demo]

## Installation
---

BoxSnake relies on the Detectron2, so please use the same installation process as Detectron2 (more ways can be found [here](https://detectron2.readthedocs.io/tutorials/install.html)):
``` shell
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

BoxSnake also uses the deformable attention modules introduced in [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) and the differentiable rasterizer introduced in [BoundaryFormer](https://github.com/mlpc-ucsd/BoundaryFormer). Please build them on your system:
``` shell
bash scripts/auto_build.sh
```
or 
``` shell
cd ./modeling/layers/deform_attn
sh ./make.sh
cd ./modeling/layers/diff_ras
python setup.py build install
```

## Model Zoo
----

### COCO

| Arch |  Backbone  | lr<br>sched | mask <br>AP | Download |
|:----:|:----------:|:-----------:|:-----------:|:--------:|
| RCNN |   R50-FPN  |      1X     |     31.1    |  weights |
| RCNN |  R101-FPN  |      2X     |     31.6    |  weights |
| RCNN |  R101-FPN  |      1X     |     31.6    |  weights |
| RCNN |  R101-FPN  |      2X     |     32.1    |  weights |
| RCNN | Swin-B-FPN |      1X     |     38.3    |  weights |
| RCNN | Swin-L-FPN |      1X     |     38.9    |  weights |

mask AP is the result on validation set.

### Cityscapes

| Arch | Backbone | lr<br>sched | mask <br>AP | Download |
|:----:|:--------:|:-----------:|:-----------:|:--------:|
| RCNN |  R50-FPN |   24K iter  |     26.3    |  weights |


## Getting Start
----
We use the [COCO dataset](https://cocodataset.org/#home) and [Cityscapes dataset](https://www.cityscapes-dataset.com). Please following [here](https://github.com/Yangr116/BoxSnake/datasets/README.md) to prepare them.

If you would like to use swin transformer backbone, please download swin weights from [here](https://github.com/microsoft/Swin-Transformer) and convert them to pkl format:
```
python tools/convert-pretrained-model-to-d2.py ${your_swin_pretrained.pth} ${yout_swin_pretrained.pkl}
``` 

### Training
To train on COCO dataset using the R50 backbone at a 1X schedule:
```shell
# 8 gpus
python train_net.py --num-gpus 8 --config-file configs/COCO-InstanceSegmentation/BoxSnake_RCNN/boxsnake_rcnn_R_50_FPN_1x.yaml
```

You can also run below code:
```
bash scripts/auto_run.sh $CONFIG  # your config
```

### Inference
To inference on COCO validation set using trained weights:
```shell
# 8 gpus
python train_net.py --num-gpus 8 --config-file configs/COCO-InstanceSegmentation/BoxSnake_RCNN/boxsnake_rcnn_R_50_FPN_1x.yaml
 --eval-only MODEL.WEIGHTS ${your/checkpoints/boxsnake_rcnn_R_50_FPN_coco_1x.pth}
```
Inference on a single image using trained weights:
```shell
python demo/demo.py --config-file configs/COCO-InstanceSegmentation/BoxSnake_RCNN/boxsnake_rcnn_R_50_FPN_1x.yaml --input demo/demo.jpg --output ${/your/visualized/dir} --confidence-threshold 0.5 --opts MODEL.WEIGHTS ${your/checkpoints/boxsnake_rcnn_R_50_FPN_coco_1x.pth}
```

## Others
---

BoxSnake is inspired by traditional levelset and GVF methods, and you can check below links to learn them:

- https://www.csd.uwo.ca/~yboykov/Presentations/ECCV06_tutorial_partII_dan.pdf
- https://www.youtube.com/watch?v=1ZJ88JyLPZI
- https://agustinus.kristia.de/techblog/2016/11/05/levelset-method
- https://agustinus.kristia.de/techblog/2016/11/20/levelset-segmentation  

Some geometric knowledge may help readers to understand the BoxSnake better:

- Points in Polygon (PIP): https://wrfranklin.org/Research/Short_Notes/pnpoly.html#The%20Method
- PIP: https://towardsdatascience.com/is-the-point-inside-the-polygon-574b86472119
- PIP numpy: https://github.com/Dan-Patterson/numpy_geometry/blob/dbc1a00baaf86d8ae437236659a45cfb57f4f35e/arcpro_npg/npg/npg/npg_pip.py
- Distance from a point to a line: http://paulbourke.net/geometry/pointlineplane
- Distance from a point to a polygon: https://stackoverflow.com/questions/10983872/distance-from-a-point-to-a-polygon


## Acknowledgement

If you find BoxSnake helpful, please cite:
```
@misc{BoxSnake,
      title={BoxSnake: Polygonal Instance Segmentation with Box Supervision}, 
      author={Rui Yang and Lin Song and Yixiao Ge and Xiu Li},
      year={2023},
      eprint={2303.11630},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```