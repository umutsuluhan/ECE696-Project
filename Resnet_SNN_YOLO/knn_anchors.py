import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from dataset import VisdroneDataset
from torch.utils.data import DataLoader

def iou_matrix(anchors, gt_boxes):
    result = []
    for gt in gt_boxes:
        inter = np.minimum(gt, anchors).prod(axis=1)
        union = (gt[0] * gt[1]) + (anchors[:, 0] * anchors[:, 1]) - inter
        ious = inter / union
        result.append(ious)
    return np.array(result) 

def grid_based_anchor_avg(all_bbox_dims, width_step=0.075, height_step=0.15, max_width=0.35, max_height=0.5):
    anchors = []
    w_bins = int(np.ceil(max_width / width_step))
    h_bins = int(np.ceil(max_height / height_step))

    for wi in range(w_bins):
        for hi in range(h_bins):
            w_min = wi * width_step
            w_max = (wi + 1) * width_step
            h_min = hi * height_step
            h_max = (hi + 1) * height_step

            in_bin = (
                (all_bbox_dims[:, 0] >= w_min) & (all_bbox_dims[:, 0] < w_max) &
                (all_bbox_dims[:, 1] >= h_min) & (all_bbox_dims[:, 1] < h_max)
            )

            bin_boxes = all_bbox_dims[in_bin]
            if len(bin_boxes) > 0:
                mean_box = bin_boxes.mean(axis=0)
                anchors.append(mean_box)

    return torch.tensor(anchors, dtype=torch.float32)

def generate_anchors(train_data):
    all_bbox_dims_tensor_gpu = torch.empty(0, 2, dtype=torch.float32, device="cuda")
    for iter_idx, data in enumerate(tqdm(train_data)):
        img, targets, target_length = data
        targets = targets.to("cuda")

        non_zero_mask = (targets[:, :, 3] != 0) & (targets[:, :, 4] != 0)
        valid_widths = targets[:, :, 3][non_zero_mask].unsqueeze(1)
        valid_heights = targets[:, :, 4][non_zero_mask].unsqueeze(1)

        if valid_widths.numel() > 0:
            current_bbox_dims_gpu = torch.cat((valid_widths, valid_heights), dim=1)
            all_bbox_dims_tensor_gpu = torch.cat((all_bbox_dims_tensor_gpu, current_bbox_dims_gpu), dim=0)

    all_bbox_dims = all_bbox_dims_tensor_gpu.cpu().numpy()
    anchors = grid_based_anchor_avg(all_bbox_dims, width_step=0.075, height_step=0.15)
    all_boxes = all_bbox_dims_tensor_gpu.cpu().numpy()
    plt.scatter(all_boxes[:, 0], all_boxes[:, 1], alpha=0.3, label="GT Boxes")
    plt.scatter(anchors[:, 0], anchors[:, 1], marker="x", color="red", label="Anchors")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.legend()
    plt.title("Anchor vs Ground Truth BBox Distribution")
    plt.savefig("anchor_dist.png")

    ious = iou_matrix(anchors.numpy(), all_bbox_dims)
    max_ious = ious.max(axis=1)
    print("GT boxes with IoU > 0.5:", np.mean(max_ious > 0.5) * 100, "%")

    return anchors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training configuration.")
    parser.add_argument('--batch_size', type=int, default=128, help='training dataset batch size')
    args = parser.parse_args()

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    dataset_train = VisdroneDataset("train", "cuda", num_steps=1, stride=1)
    train_data = DataLoader(dataset_train, batch_size=128, drop_last=True, shuffle=True)

    anchors = generate_anchors(train_data)
    print(anchors)
