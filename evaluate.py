import numpy as np

import box_utils

# Didn't match them by image
# def calculate_precision_recall(pred, gt, iou_threshold=0.5):
#     # true_positives = 0
#     # false_positives = 0
#     # false_negatives = 0

#     # for pred_box in pred:
#     #     ious = box_utils.calculate_iou(np.array([pred_box[1]]), np.array([g[1] for g in gt]))
#     #     max_iou = np.max(ious)

#     #     if max_iou >= iou_threshold:
#     #         true_positives += 1
#     #     else:
#     #         false_positives += 1

#     # false_negatives = max(len(gt) - true_positives, 0)

#     ious = box_utils.calculate_iou(np.array(pred), np.array(gt))
#     max_ious = np.max(ious, axis=-1)

#     true_positives = np.count_nonzero(max_ious >= iou_threshold)
#     false_positives = len(max_ious) - true_positives
#     false_negatives = np.maximum(len(gt) - true_positives, 0)

#     precision = true_positives / (true_positives + false_positives + 1e-10)
#     recall = true_positives / (true_positives + false_negatives + 1e-10)

#     return precision, recall


# def calculate_average_precision(predictions, ground_truth, iou_threshold=0.5):
#     precision_values = []
#     recall_values = []

#     for confidence_threshold in np.arange(1, -0.1, -0.1):  # Typically this would rather be np.arange(0, 1.1, 0.1)
#         confident_boxes = [pred[1] for pred in predictions if pred[2] >= confidence_threshold] or np.empty(shape=(0, 4))
#         gt_boxes = [gt[1] for gt in ground_truth]

#         precision, recall = calculate_precision_recall(confident_boxes, gt_boxes, iou_threshold)
#         precision_values.append(precision)
#         recall_values.append(recall)

#     precision_values = np.array(precision_values)
#     recall_values = np.array(recall_values)

#     interpolated_precision = np.maximum.accumulate(precision_values[::-1])[::-1]  # [np.max(precision_values[i:]) for i in range(len(precision_values))]
#     recall_diff = np.diff(recall_values, prepend=0)

#     average_precision = np.sum(interpolated_precision * recall_diff)

#     return average_precision


# def evaluate(model, dataset, labels):
#     all_preds = []
#     all_gts = []

#     for image, gt_boxes, gt_labels in dataset:
#         pred = model.inference(image, labels)

#         # box_utils.plot_ious(gt_boxes, pred[1], image, labels=pred[0], confidences=pred[2])

#         all_preds.append(list(zip(*pred)))
#         all_gts.append(list(zip(*[np.array(labels)[gt_labels], gt_boxes])))

#     ap_values = []
#     for label in labels:
#         pred_class = [pred for preds in all_preds for pred in preds if pred[0] == label]
#         gt_class = [gt for gts in all_gts for gt in gts if gt[0] == label]

#         average_precision = calculate_average_precision(pred_class, gt_class)
#         ap_values.append(average_precision)

#     mean_average_precision = np.mean(ap_values)

#     return mean_average_precision


def calculate_precision_recall(preds_gts, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # for pred_box in pred:
    #     ious = box_utils.calculate_iou(np.array([pred_box[1]]), np.array([g[1] for g in gt]))
    #     max_iou = np.max(ious)

    #     if max_iou >= iou_threshold:
    #         true_positives += 1
    #     else:
    #         false_positives += 1

    # false_negatives = max(len(gt) - true_positives, 0)
    for pred, gt in preds_gts:
        if len(pred) and len(gt):
            ious = box_utils.calculate_iou(pred, gt)
            max_ious = np.max(ious, axis=-1)

            amount_pos = np.count_nonzero(max_ious >= iou_threshold)
        else:
            amount_pos = 0

        true_positives += amount_pos
        false_positives += len(pred) - amount_pos
        false_negatives += np.maximum(len(gt) - amount_pos, 0)

    # ious = box_utils.calculate_iou(np.array([pr[0][1] for pr in pred]), np.array([g[0][1] for g in gt]))
    # max_ious = np.max(ious, axis=-1)

    # true_positives = np.count_nonzero(max_ious >= iou_threshold)
    # false_positives = len(max_ious) - true_positives
    # false_negatives = np.maximum(len(gt) - true_positives, 0)

    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)

    return precision, recall


def calculate_average_precision(preds_gts, iou_threshold=0.5):
    precision_values = []
    recall_values = []

    for confidence_threshold in np.arange(1, -0.1, -0.1):
        confident_preds_gts = []
        for preds, gts in preds_gts:
            pr = np.array([pred[1] for pred in preds if pred[2] >= confidence_threshold])
            g = np.array([gt[1] for gt in gts])

            if not len(pr):
                pr = np.empty(shape=(0, 4))
            if not len(g):
                g = np.empty(shape=(0, 4))
            if len(pr) or len(g):
                confident_preds_gts.append((pr, g))
        # confident_preds_gts = [([pr for pr in pred_gt[0] if pr[2] >= 0.1] or empty, pred_gt[1] or empty) for pred_gt in preds_gts]
        # reduced_confident_preds_gts = [pred_gt for pred_gt in confident_preds_gts if pred_gt != ([["", np.empty(4,)]], [["", np.empty((4,))]])]

        precision, recall = calculate_precision_recall(confident_preds_gts, iou_threshold)
        precision_values.append(precision)
        recall_values.append(recall)

    precision_values = np.array(precision_values)
    recall_values = np.array(recall_values)

    interpolated_precision = np.maximum.accumulate(precision_values[::-1])[::-1]  # [np.max(precision_values[i:]) for i in range(len(precision_values))]
    recall_diff = np.diff(recall_values, prepend=0)

    average_precision = np.sum(interpolated_precision * recall_diff)

    return average_precision


def evaluate(model, dataset, labels):
    all_preds_gts = []

    for image, gt_boxes, gt_labels in dataset:
        pred = model.inference(image, labels)

        # box_utils.plot_ious(gt_boxes, pred[1], image, labels=pred[0], confidences=pred[2])

        preds = list(zip(*pred))
        gts = list(zip(*[np.array(labels)[gt_labels], gt_boxes]))
        all_preds_gts.append((preds, gts))

    ap_values = []
    for label in labels:
        preds_gts = [([pred for pred in preds if pred[0] == label], [gt for gt in gts if gt[0] == label]) for preds, gts in all_preds_gts]
        reduced_preds_gts = [pred_gt for pred_gt in preds_gts if pred_gt != ([], [])]

        average_precision = calculate_average_precision(reduced_preds_gts)
        ap_values.append(average_precision)

    mean_average_precision = np.mean(ap_values)

    return mean_average_precision
