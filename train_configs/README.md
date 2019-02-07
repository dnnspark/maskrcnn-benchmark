### From laptop

Only `IMS_PER_BATCH=1` works, but with following restrictions:

- Learning rate and number of iterations in learning schedule must be changed accordingly.
- There is significant typing lag.
```
python tools/train_net.py --config-file "train_configs/e2e_faster_rcnn_R_50_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 1440000 SOLVER.STEPS "(960000, 1280000)"
```

All attempts for using large batch size failed with OOM error, E.g:
```
RuntimeError: CUDA out of memory. Tried to allocate 49.00 MiB (GPU 0; 3.95 GiB total capacity; 2.43 GiB already allocated; 17.81 MiB free; 81.24 MiB cached)
```

```
python tools/train_net.py --config-file "train_configs/e2e_faster_rcnn_R_50_FPN_1x.yaml" OUTPUT_DIR "./out/frcnn"
python tools/train_net.py --config-file "train_configs/e2e_faster_rcnn_R_50_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)"
```

### From `p3.x16large` instance (8 gpus)

Faster RCNN:
```
python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file "train_configs/e2e_faster_rcnn_R_50_FPN_1x.yaml" OUTPUT_DIR "./out/frcnn"
```

Mask RCNN:
```
python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file "train_configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" OUTPUT_DIR "./out/maskrcnn"
```
