import torch
from maskrcnn_benchmark.layers.roi_align import _ROIAlign
from maskrcnn_benchmark.layers import nms as _box_nms



def get_random_boxes(num_boxes):
    H,W = 480, 640
    x1 = torch.rand(num_boxes) * 0.7*W
    y1 = torch.rand(num_boxes) * 0.7*H
    w = torch.rand(num_boxes) * 0.5*W
    h = torch.rand(num_boxes) * 0.5*H
    x2,y2 = x1+w, y1+h

    return torch.stack([x1,y1,x2,y2], dim=1)


def test_nms():
    boxes = get_random_boxes(1000)
    scores = torch.randn(1000)
    nms_thresh = .7 

    keep_cpu = _box_nms(boxes, scores, nms_thresh) 
    keep_gpu = _box_nms(boxes.cuda(), scores.cuda(), nms_thresh) 

    assert torch.allclose(keep_cpu, keep_gpu.cpu())
    print("test_nms: OK")

    return;

roi_align = _ROIAlign.apply

def test_roi_align():
    input = torch.randn(1,256,200,272) * 8.
    rois = torch.cat([torch.zeros(1000,1), get_random_boxes(1000)], dim=1)
    output_size = (7,7)
    spatial_scale = .25
    sampling_ratio = 2

    aligned_cpu = roi_align(input, rois, output_size, spatial_scale, sampling_ratio)
    aligned_gpu = roi_align(input.cuda(), rois.cuda(), output_size, spatial_scale, sampling_ratio)

    # assert torch.allclose(aligned_cpu, aligned_gpu.cpu(), atol=0.0005)
    if not torch.allclose(aligned_cpu, aligned_gpu.cpu()):
        max_diff = torch.abs(aligned_cpu - aligned_gpu.cpu()).max()
        print('test_roi_align: error=%.6f' % max_diff) 
    else:
        print("test_roi_align: OK")

    return


if __name__ == '__main__':
    test_nms()
    test_roi_align()