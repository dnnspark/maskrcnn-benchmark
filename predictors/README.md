###  Running a trained model on all images in a folder sequentially.
```
python predictors/run_on_image_folder.py --config-file "test_configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" --checkpoint-file "out/maskrcnn/model_final.pth" --image-folder "/cluster/storage/vdata/datasets/coco/images/val2017/" --show-mask-heatmaps
```
