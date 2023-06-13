# Prepare Datasets

We use the same steps as the Detectron2 to prepare dataset, so you can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.

## Expected dataset structure for [COCO](https://cocodataset.org/#download):

```
coco/
  annotations/
    instances_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```


## Expected dataset structure for [cityscapes](https://www.cityscapes-dataset.com/downloads/):
```
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
  leftImg8bit/
    train/
    val/
    test/
```
Install cityscapes scripts by:
```
pip install git+https://github.com/mcordts/cityscapesScripts.git
```
