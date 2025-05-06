import os
import cv2
import copy
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import snntorch.functional as SF

from tqdm import tqdm
from snn_model import SNNYOLO
from dataset import VisdroneDataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

num_steps = 10

anchors = torch.tensor ([[0.0257, 0.0472],
                        [0.0529, 0.1752],
                        [0.0951, 0.0866],
                        [0.1064, 0.1911],
                        [0.1080, 0.3361],
                        [0.1366, 0.4577],
                        [0.1666, 0.1202],
                        [0.1818, 0.1969],
                        [0.1978, 0.3398],
                        [0.2627, 0.1275],
                        [0.2443, 0.2308],
                        [0.2371, 0.3190],
                        [0.3430, 0.4407],
                        [0.3430, 0.5210]])

def yolo_loss(class_pred, bbox_pred, obj_pred, targets, anchors, num_classes, target_lengths):
    # Get batch and grid parameters.
    batch_size, grid_size, _, num_anchors, _ = class_pred.shape
    device = class_pred.device

    # Accumulate total losses and metrics over the batch.
    total_loss = 0.0
    total_bbox_loss = 0.0
    total_cls_loss = 0.0
    total_conf_loss = 0.0
    total_iou_sum = 0.0
    total_positive = 0  # Count of positive (object) anchors

    # Loss functions.
    bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1, reduction="mean")
    mse_loss = nn.MSELoss(reduction="mean")

    # YOLO loss weight factors.
    lambda_coord = 10.0   # Weight for bbox regression loss for cells with objects.
    lambda_noobj = 0.5   # Weight for confidence loss for cells with no object.

    # print(target_lengths)

    best_ious = []

    # Process each sample in the batch.
    for b in range(batch_size):
        # Create target tensors matching the shape of model outputs.
        obj_target   = torch.zeros((grid_size, grid_size, num_anchors), device=device)
        bbox_target  = torch.zeros((grid_size, grid_size, num_anchors, 4), device=device)
        # For classification targets, we use a tensor with integer class labels.
        class_target = torch.zeros((grid_size, grid_size, num_anchors), dtype=torch.long, device=device)
        # A mask to indicate the anchors assigned to a ground truth.
        obj_mask = torch.zeros((grid_size, grid_size, num_anchors), dtype=torch.bool, device=device)

        # Get the valid targets for this sample. Each sample has padded targets.
        num_valid = int(target_lengths[b].item()) if torch.is_tensor(target_lengths[b]) else int(target_lengths[b])
        sample_targets = targets[b]  # shape: [max_targets, 5]

        # Loop over each valid target for this sample.
        for t in range(num_valid):
            # Each target is represented as: [class, x_center, y_center, width, height].
            cls = int(sample_targets[t, 0].item())
            x   = sample_targets[t, 1].item()
            y   = sample_targets[t, 2].item()
            w   = sample_targets[t, 3].item()
            h   = sample_targets[t, 4].item()


            # Determine the grid cell location.
            grid_x = min(int(x * grid_size), grid_size - 1)
            grid_y = min(int(y * grid_size), grid_size - 1)
            # Compute offsets relative to the top-left corner of the grid cell.
            cell_x = x * grid_size - grid_x
            cell_y = y * grid_size - grid_y

            # Determine the best matching anchor using IoU computed on box widths/heights.
            best_iou = 0.0
            best_anchor = 0
            for a in range(num_anchors):
                anchor_w, anchor_h = anchors[a]
                # Compute intersection area based on widths/heights.
                inter_w = min(w, anchor_w)
                inter_h = min(h, anchor_h)
                intersection = inter_w * inter_h
                union = w * h + anchor_w * anchor_h - intersection
                iou = intersection / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = a

            best_ious.append(best_iou.item())
            # Assign this target to the grid cell and anchor.
            obj_mask[grid_y, grid_x, best_anchor] = True
            obj_target[grid_y, grid_x, best_anchor] = 1.0
            bbox_target[grid_y, grid_x, best_anchor, 0] = cell_x
            bbox_target[grid_y, grid_x, best_anchor, 1] = cell_y
            bbox_target[grid_y, grid_x, best_anchor, 2] = w
            bbox_target[grid_y, grid_x, best_anchor, 3] = h
            class_target[grid_y, grid_x, best_anchor] = cls

        # Compute objectness (confidence) loss for this sample.
        conf_loss = bce_loss(obj_pred[b], obj_target)

        if obj_mask.any():
            conf_pos = bce_loss(obj_pred[b][obj_mask],
                                obj_target[obj_mask])
        else:
            conf_pos = torch.tensor(0., device=device)

        # Negative objectness loss (weighted)
        if (~obj_mask).any():
            conf_neg = bce_loss(obj_pred[b][~obj_mask],
                                obj_target[~obj_mask])
            conf_neg = conf_neg
        else:
            conf_neg = torch.tensor(0., device=device)

        conf_loss = (conf_pos + conf_neg) / obj_pred[b][obj_mask].shape[0]

        # Compute the bbox regression and classification losses only for the positive anchors.
        if obj_mask.sum() > 0:
            sample_bbox_loss = mse_loss(bbox_pred[b][obj_mask], bbox_target[obj_mask]) / obj_pred[b][obj_mask].shape[0]
            sample_cls_loss  = ce_loss(class_pred[b][obj_mask], class_target[obj_mask]) / obj_pred[b][obj_mask].shape[0]
        else:
            sample_bbox_loss = torch.tensor(0.0, device=device)
            sample_cls_loss  = torch.tensor(0.0, device=device)

        # Total loss for this sample.
        sample_loss = lambda_coord * sample_bbox_loss + sample_cls_loss + lambda_noobj * conf_loss

        # Accumulate losses.
        total_loss += sample_loss 
        total_bbox_loss += lambda_coord * sample_bbox_loss
        total_cls_loss += sample_cls_loss 
        total_conf_loss += lambda_noobj * conf_loss

        # --- Compute IoU for the positive (object) predictions ---
        # Helper to convert a box from grid coordinates to image (normalized) coordinates.
        def convert_box(i, j, box):
            abs_x = (j + box[0]) / grid_size
            abs_y = (i + box[1]) / grid_size
            # The box remains [abs_center_x, abs_center_y, width, height]
            return torch.tensor([abs_x, abs_y, box[2], box[3]], device=device)

        # Helper to compute IoU between two boxes provided in (cx, cy, w, h) format.
        def compute_iou(boxA, boxB):
            # Convert box from center-based representation to (x1, y1, x2, y2)
            def to_corners(box):
                cx, cy, w, h = box[0], box[1], box[2], box[3]
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                return x1, y1, x2, y2
            ax1, ay1, ax2, ay2 = to_corners(boxA)
            bx1, by1, bx2, by2 = to_corners(boxB)
            # Calculate intersection.
            inter_x1 = max(ax1, bx1)
            inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)
            inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)
            areaA = (ax2 - ax1) * (ay2 - ay1)
            areaB = (bx2 - bx1) * (by2 - by1)
            union_area = areaA + areaB - inter_area + 1e-6
            return inter_area / union_area

        # Compute IoU for each positive anchor.
        pos_indices = obj_mask.nonzero(as_tuple=False)
        iou_sum_sample = 0.0
        count_positive = 0
        # print(len(pos_indices))
        for idx in pos_indices:
            i, j, a = idx[0].item(), idx[1].item(), idx[2].item()
            pred_box = convert_box(i, j, bbox_pred[b, i, j, a])
            targ_box = convert_box(i, j, bbox_target[i, j, a])
            # print(pred_box, targ_box)
            iou_val = compute_iou(pred_box, targ_box)
            iou_sum_sample += iou_val
            count_positive += 1

        total_iou_sum += iou_sum_sample
        total_positive += count_positive

        # print(total_positive)
        # exit()
    # Average IoU across all positive predictions in the batch.
    if total_positive > 0:
        avg_iou = total_iou_sum / total_positive
    else:
        avg_iou = torch.tensor(0.0, device=device)

    return total_loss, avg_iou, total_bbox_loss, total_cls_loss, total_conf_loss

def validation(val_data, model, anchors, num_classes):
    model.eval()
    total_loss = 0.0
    total_bbox_loss = 0.0
    total_cls_loss = 0.0
    total_conf_loss = 0.0
    total_iou = 0.0
    
    # Regular validation loop
    with torch.no_grad():
        for valIndex, data in enumerate(tqdm(val_data)):
            img, targets, target_length = data
            img = img.to("cuda")
            targets = targets.to("cuda")
            target_length = target_length.to("cuda")

            spk_rec, mem_rec = model(img, num_steps=num_steps)

            last_mem = mem_rec[-1]
            class_pred = last_mem[..., :num_classes]
            bbox_pred  = torch.sigmoid(last_mem[..., num_classes:num_classes+4])
            obj_pred   = last_mem[..., num_classes+4]

            loss, avg_iou, bbox_loss, cls_loss, conf_loss = yolo_loss(
                class_pred, bbox_pred, obj_pred,
                targets, anchors, num_classes, target_length
            )
            
            total_loss += loss.item()
            total_iou += avg_iou
            total_bbox_loss += bbox_loss
            total_cls_loss += cls_loss
            total_conf_loss += conf_loss

    model.train()
    avg_loss = total_loss / (valIndex + 1) if (valIndex + 1) > 0 else 0
    avg_val_iou = total_iou / (valIndex + 1) if (valIndex + 1) > 0 else 0
    avg_bbox_loss = total_bbox_loss / (valIndex + 1) if (valIndex + 1) > 0 else 0
    avg_cls_loss = total_cls_loss / (valIndex + 1) if (valIndex + 1) > 0 else 0
    avg_conf_loss = total_conf_loss / (valIndex + 1) if (valIndex + 1) > 0 else 0
    return avg_loss, avg_val_iou, avg_bbox_loss, avg_cls_loss, avg_conf_loss

def train_model(model, train_data, val_data, weights_path, writer, log_file, num_epochs=10, learning_rate=0.01, num_classes=10):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    model.train()
    loss_report_interval = 10
    val_interval = max(1, int(len(train_data) - 1))
    val_interval = 1

    log_file.write(f"Training model with {len(train_data)} training samples and {len(val_data)} validation samples.\n")
    print(f"Loss report interval: {loss_report_interval}, Validation interval: {val_interval}")
    val_step = 0

    start_time = time.time()

    # Regular training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_bbox_loss = 0.0
        total_cls_loss = 0.0
        total_conf_loss = 0.0
        total_iou = 0.0
        train_step = 10
        val_step += 1

        for trainIndex, data in enumerate(train_data):
            # print("Train Index:", trainIndex)
            img, targets, target_length = data
            img = img.to("cuda")
            targets = targets.to("cuda")
            target_length = target_length.to("cuda")

            optimizer.zero_grad()
            spk_rec, mem_rec = model(img, num_steps=num_steps)

            loss_val = 0
            for (spk, mem) in zip(spk_rec, mem_rec):
                class_pred = mem[..., :num_classes]
                bbox_pred = torch.sigmoid(mem[..., num_classes:num_classes + 4])
                obj_pred = mem[..., num_classes + 4]

                loss, avg_iou, bbox_loss, cls_loss, conf_loss = yolo_loss(class_pred, bbox_pred, obj_pred, targets, anchors, num_classes, target_length)

                total_loss += loss.item() / num_steps
                total_iou += avg_iou  / num_steps
                total_bbox_loss += bbox_loss.item()   / num_steps
                total_cls_loss += cls_loss.item()   / num_steps
                total_conf_loss += conf_loss.item()   / num_steps

                loss_val += loss / num_steps

            total_loss += loss_val.item()
            total_iou += total_iou / len(mem_rec)

            loss_val.backward()
            optimizer.step()

            last_mem = mem_rec[-1]
            class_pred = last_mem[..., :num_classes]
            bbox_pred  = torch.sigmoid(last_mem[..., num_classes:num_classes+4])
            obj_pred   = last_mem[..., num_classes+4]

            loss, avg_iou, bbox_loss, cls_loss, conf_loss = yolo_loss(
                class_pred, bbox_pred, obj_pred,
                targets, anchors, num_classes, target_length
            )

            total_loss += loss.item()
            total_iou += avg_iou
            total_bbox_loss += bbox_loss.item()
            total_cls_loss += cls_loss.item()
            total_conf_loss += conf_loss.item()

            end_time = time.time()

            if (trainIndex + 1) % loss_report_interval == 0:
                avg_loss = total_loss / (trainIndex + 1)
                avg_training_iou = total_iou / (trainIndex + 1)
                avg_bbox_loss = total_bbox_loss / (trainIndex + 1)
                avg_cls_loss = total_cls_loss / (trainIndex + 1)
                avg_conf_loss = total_conf_loss / (trainIndex + 1)
                avg_time_per_batch = (end_time - start_time) / (trainIndex + 1)
                log_file.write(f"Epoch {epoch+1}/{num_epochs}, Iteration: {trainIndex+1}/{len(train_data)}, Loss: {avg_loss:.4f}, IoU: {avg_training_iou:.4f}\n")
                print(f"""Epoch {epoch+1}/{num_epochs}, 
                Iteration: {trainIndex+1}/{len(train_data)}, 
                Loss: {avg_loss:.4f}, 
                BBox Loss: {avg_bbox_loss:.4f},
                Class Loss: {avg_cls_loss:.4f},
                Conf Loss: {avg_conf_loss:.4f},
                Average Time Per Batch: {avg_time_per_batch:.4f}""")
                train_step += 10

            if (trainIndex + 1) % val_interval == 0 and trainIndex != 0:
                print("Validating...")
                avg_val_loss, avg_val_iou, avg_bbox_loss, avg_cls_loss, avg_conf_loss = validation(val_data, model, anchors, num_classes)
                scheduler.step(avg_val_loss)
                log_file.write(f"Epoch {epoch+1}/{num_epochs}, Iteration: {trainIndex+1}/{len(train_data)}, Val Loss: {avg_val_loss:.4f}, Val IOU: {avg_val_iou:.4f}\n")
                print(f"""
                Epoch {epoch+1}/{num_epochs}, 
                Iteration: {trainIndex+1}/{len(train_data)}, 
                Val Loss: {avg_val_loss:.4f}, 
                BBox Loss: {avg_bbox_loss:.4f},
                Class Loss: {avg_cls_loss:.4f},
                Conf Loss: {avg_conf_loss:.4f},
                Val IOU: {avg_val_iou:.4f}""")

        torch.save(model.state_dict(), f"{weights_path}snn_model_epoch_{epoch+1}.pth")
        torch.save(optimizer.state_dict(), f"{weights_path}snn_optimizer_epoch_{epoch+1}.pth")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training configuration.")
    parser.add_argument('--batch_size', type=int, default=32, help='training dataset batch size')
    args = parser.parse_args()

    weights_path = "./weights/"
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    log_file = open("snn_log.txt", "w")

    model = SNNYOLO(num_classes=10, num_boxes=14).to("cuda")
    log_file.write(f"Using SNN_YOLO model.\n")
    log_file.write(f'Total trainable parameters: {count_parameters(model)}\n')
    print("Using SNN_YOLO model.")
    print(f'Total trainable parameters: {count_parameters(model)}')

    dataset_train = VisdroneDataset("train", "cuda", 10, 5)
    train_data = DataLoader(dataset_train, batch_size=args.batch_size, drop_last=True, shuffle=True)

    dataset_val = VisdroneDataset("val", "cuda", 10, 5)
    val_data = DataLoader(dataset_val, batch_size=args.batch_size, drop_last=True, shuffle=False)

    log_dir = "./runs"
    writer = SummaryWriter(log_dir=log_dir)

    log_file.write(f"Training model with {len(train_data)} training samples and {len(val_data)} validation samples.\n")
    print(f"Training model with {len(train_data)} training samples and {len(val_data)} validation samples.")

    train_model(model, train_data, val_data, weights_path, writer, log_file)

    log_file.close()
    writer.close()