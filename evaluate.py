import numpy as np
import matplotlib.pyplot as plt

import box_utils
import config

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

#     pred_boxes, pred_confs = zip(*[pred[1:] for pred in predictions])
#     gt_boxes = np.array([gt[1] for gt in ground_truth])
#     for confidence_threshold in np.arange(1, -0.1, -0.1):  # np.arange(0, 1.1, 0.01):
#         confident_boxes = np.array(pred_boxes)[pred_confs >= confidence_threshold]

#         precision, recall = calculate_precision_recall(confident_boxes, gt_boxes, iou_threshold)
#         precision_values.append(precision)
#         recall_values.append(recall)

#     precision_values = np.array(precision_values)
#     recall_values = np.array(recall_values)

#     interpolated_precision = np.maximum.accumulate(precision_values[::-1])[::-1]  # [np.max(precision_values[i:]) for i in range(len(precision_values))]
#     recall_diff = np.diff(recall_values, prepend=0)

#     average_precision = np.sum(interpolated_precision * recall_diff)

#     return average_precision, precision_values, recall_values


# def evaluate(model, dataset, labels):
#     all_preds = []
#     all_gts = []

#     for image, gt_boxes, gt_labels in dataset:
#         pred = model.inference(image, labels, confidence_threshold=0.05)

#         # box_utils.plot_ious(gt_boxes, pred[1]x, image, labels=pred[0], confidences=pred[2])

#         all_preds.append(list(zip(*pred)))
#         all_gts.append(list(zip(*[np.array(labels)[gt_labels], gt_boxes])))

#     ap_values = []
#     precision_values = []
#     recall_values = []
#     for label in labels:
#         pred_class = [pred for preds in all_preds for pred in preds if pred[0] == label]
#         gt_class = [gt for gts in all_gts for gt in gts if gt[0] == label]

#         average_precision, precisions, recalls = calculate_average_precision(pred_class, gt_class)
#         ap_values.append(average_precision)
#         precision_values.append(precisions)
#         recall_values.append(recalls)

#     mean_average_precision = np.mean(ap_values)

#     plot_p_r(precision_values, recall_values, ap_values, labels)

#     return mean_average_precision


def plot_prec_rec(precision_values, recall_values, ap_values, labels):
    _, ax = plt.subplots(figsize=(10, 10))

    mAP = np.mean(ap_values)

    for precision_value, recall_value, ap_value, label in zip(precision_values, recall_values, ap_values, labels):
        ax.plot(recall_value, precision_value, label=f"{label} (AP: {(ap_value * 100):.2f}%)")
        ax.set_xscale("linear")
        ax.legend()

    plt.title(f"Precision-Recall Curve ({len(labels)} classes, mAP: {(mAP * 100):.2f}%)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    # plt.grid()

    margins = 0.05
    plt.xlim(-margins, 1 + margins)
    plt.ylim(-margins, 1 + margins)

    plt.savefig(f"{config.SAVE_FOLDER_PATH}/precision_recall_curve.png")


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

    return average_precision, precision_values, recall_values


def evaluate(model, dataset, labels):
    all_preds_gts = []

    for image, gt_boxes, gt_labels in dataset:
        pred = model.inference(image, labels, confidence_threshold=0.05)  # Supposed to be 0.01 (as in SSD paper)

        # box_utils.plot_ious(gt_boxes, pred[1], image, labels=pred[0], confidences=pred[2])

        preds = list(zip(*pred))
        gts = list(zip(*[np.array(labels)[gt_labels], gt_boxes]))
        all_preds_gts.append((preds, gts))

    ap_values = []
    precision_values = []
    recall_values = []
    for label in labels:
        preds_gts = [([pred for pred in preds if pred[0] == label], [gt for gt in gts if gt[0] == label]) for preds, gts in all_preds_gts]
        reduced_preds_gts = [pred_gt for pred_gt in preds_gts if pred_gt != ([], [])]

        average_precision, precisions, recalls = calculate_average_precision(reduced_preds_gts)
        ap_values.append(average_precision)
        precision_values.append(precisions)
        recall_values.append(recalls)

    mean_average_precision = np.mean(ap_values)  # What about if there are no gts for the class?

    plot_prec_rec(precision_values, recall_values, ap_values, labels)

    return mean_average_precision


# def calculate_precision_recall(preds, gts, iou_threshold=0.5):
#     if len(preds) and len(gts):
#         ious = box_utils.calculate_iou(preds, gts)
#         max_ious = np.max(ious, axis=-1)

#         amount_pos = np.count_nonzero(max_ious >= iou_threshold)
#     else:
#         amount_pos = 0

#     true_positives = amount_pos
#     false_positives = len(preds) - amount_pos
#     false_negatives = np.maximum(len(gts) - amount_pos, 0)

#     return true_positives, false_positives, false_negatives


# def calculate_average_precision(preds, confs, gts, iou_threshold=0.5):
#     true_pos = []
#     false_pos = []
#     false_neg = []

#     for confidence_threshold in np.arange(1, -0.1, -0.1):
#         confident_preds = preds[confs >= confidence_threshold]

#         true_positives, false_positives, false_negatives = calculate_precision_recall(confident_preds, gts, iou_threshold)
#         true_pos.append(true_positives)
#         false_pos.append(false_positives)
#         false_neg.append(false_negatives)

#     return true_pos, false_pos, false_neg


# def evaluate(model, dataset, labels):
#     true_pos = np.zeros(shape=(len(labels), 11))
#     false_pos = np.zeros(shape=(len(labels), 11))
#     false_neg = np.zeros(shape=(len(labels), 11))

#     for image, gt_boxes, gt_labels in dataset:
#         pred = model.inference(image, labels, confidence_threshold=0.05)

#         # box_utils.plot_ious(gt_boxes, pred[1], image, labels=pred[0], confidences=pred[2])

#         ap_values = []

#         pred_labels = [labels.index(label) for label in pred[0]]
#         for label in set(gt_labels + pred_labels):
#             gts = gt_boxes[np.where(np.array(gt_labels) == label)]
            
#             true_positives, false_positives, false_negatives = calculate_average_precision(pred[1], pred[2], gts)
#             true_pos[label] += true_positives
#             false_pos[label] += false_positives
#             false_neg[label] += false_negatives

#     precision_values = []
#     recall_values = []
#     for label_true_pos, label_false_pos, label_false_neg in zip(true_pos, false_pos, false_neg):
#         precision_values.append([])
#         recall_values.append([])

#         for threshold_true_pos, threshold_false_pos, threshold_false_neg in zip(label_true_pos, label_false_pos, label_false_neg):
#             precision = threshold_true_pos / (threshold_true_pos + threshold_false_pos + 1e-10)
#             recall = threshold_true_pos / (threshold_true_pos + threshold_false_neg + 1e-10)

#             precision_values[-1].append(precision)
#             recall_values[-1].append(recall)

#         prec_vals = np.array(precision_values[-1])
#         rec_vals = np.array(recall_values[-1])

#         interpolated_precision = np.maximum.accumulate(prec_vals[::-1])[::-1]  # [np.max(precision_values[i:]) for i in range(len(precision_values))]
#         recall_diff = np.diff(rec_vals, prepend=0)

#         ap = np.sum(interpolated_precision * recall_diff)
#         ap_values.append(ap)

#     mean_average_precision = np.mean(ap_values)  # What about if there are no gts for the class?

#     plot_p_r(precision_values, recall_values, ap_values, labels)

#     return mean_average_precision
