# BoxSnake: Polygonal Instance Segmentation with Box Supervision

![Intro](./figures/BoxSnake.png)
[Paper](https://arxiv.org/pdf/2303.11630.pdf) | Code will be available soon.
### Introduciton
Box-supervised instance segmentation has gained much attention as it requires only simple box annotations instead of costly mask or polygon annotations. However, existing box-supervised instance segmentation models mainly focus on mask-based frameworks. We propose a new end-to-end training technique, termed BoxSnake, to achieve effective **polygonal instance segmentation using only box annotations for the first time**. Our method consists of two loss functions: (1) a point-based unary loss that constrains the bounding box of predicted polygons to achieve coarse-grained segmentation; and (2) a distance-aware pairwise loss that encourages the predicted polygons to fit the object boundaries.

