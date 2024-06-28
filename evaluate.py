import numpy as np
import matplotlib.pyplot as plt

import box_utils
import config
from prepare import prepare_testing

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


# def calculate_precision_recall(preds, gts, iou_threshold=0.5):  # Unfinished
#     confusion = [np.array([]) for _ in range(3)]

#     # for pred_box in pred:
#     #     ious = box_utils.calculate_iou(np.array([pred_box[1]]), np.array([g[1] for g in gt]))
#     #     max_iou = np.max(ious)

#     #     if max_iou >= iou_threshold:
#     #         true_positives += 1
#     #     else:
#     #         false_positives += 1

#     # false_negatives = max(len(gt) - true_positives, 0)
#     for pred in preds:
#         index = pred[3]

#         gt = gts[index]
#         pred = pred[1]

#         if len(gt):
#             ious = box_utils.calculate_iou(gt, pred[None])
#             max_ious = np.max(ious, axis=-1)

#             amount_pos = np.count_nonzero(max_ious >= iou_threshold)

#             if amount_pos:
#                 gts[index][np.argmax(ious, axis=-1)[0]] = [-1, -1, -1, -1]
#         else:
#             amount_pos = 0

#         confusion[0] = np.append(confusion[0], amount_pos)
#         confusion[1] = np.append(confusion[1], 1 - amount_pos)
#         confusion[2] = np.append(confusion[2], np.maximum(len(gt) - amount_pos, 0))

#     return confusion


# def calculate_average_precision(preds, gts, iou_threshold=0.5):
#     sorted_preds = sorted(preds, key=lambda x: -x[2])
#     confusion = calculate_precision_recall(sorted_preds, gts, iou_threshold)

#     precision = confusion[0] / (confusion[0] + confusion[1] + 1e-10)
#     recall = confusion[0] / (confusion[0] + confusion[2] + 1e-10)

#     precision_values = np.array(precision)
#     recall_values = np.array(recall)

#     amount_sample_points = 11
#     ap = 0
#     for recall_threshold in np.arange(0, 1.1, 1.1 / amount_sample_points):
#         prec_rec = precision_values[recall_values >= recall_threshold]
#         if len(prec_rec):
#             ap += np.max(prec_rec)
#     ap /= amount_sample_points

#     return ap, precision_values, recall_values


# def evaluate(model, dataset, labels):
#     all_preds_gts = []

#     for image, gt_boxes, gt_labels in dataset:
#         pred = model.inference(image, labels, confidence_threshold=0.05)  # Supposed to be 0.01 (as in SSD paper)

#         # box_utils.plot_ious(gt_boxes, pred[1], image, labels=pred[0], confidences=pred[2])

#         preds = list(zip(*pred))
#         gts = list(zip(*[np.array(labels)[gt_labels], gt_boxes]))
#         all_preds_gts.append((preds, gts))

#     ap_values = []
#     precision_values = []
#     recall_values = []
#     for label in labels:
#         preds = [pred + (i,) for i, (preds, _) in enumerate(all_preds_gts) for pred in preds if pred[0] == label]
#         gts = [np.array([gt[1] for gt in gts if gt[0] == label]) for _, gts in all_preds_gts]

#         average_precision, precisions, recalls = calculate_average_precision(preds, gts)
#         ap_values.append(average_precision)
#         precision_values.append(precisions)
#         recall_values.append(recalls)

#     mean_average_precision = np.mean(ap_values)  # What about if there are no gts for the class?

#     plot_prec_rec(precision_values, recall_values, ap_values, labels)

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
    plt.grid()

    margins = 0.05
    plt.xlim(-margins, 1 + margins)
    plt.ylim(-margins, 1 + margins)

    plt.savefig(f"{config.SAVE_FOLDER_PATH}/precision_recall_curve.png")
    plt.close()


"""
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

    for image_path, gt_boxes, gt_labels in dataset:
        image, gt_boxes, gt_labels = prepare_testing(image_path, gt_boxes, gt_labels, model.input_shape)
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
"""


def compute_prec_rec(tps, fps, total):
    cum_prec = np.where(tps + fps > 0, tps / (tps + fps), 0)
    cum_rec = tps / total  # cum_tp / (cum_tp[-1] + fn_tot)

    return cum_prec, cum_rec


def compute_AP_sample(prec, rec, num_recall_points=11):  # Pre-2010 sample-based AP
    ap = 0
    for t in np.linspace(0, 1, num_recall_points, endpoint=True):
        cum_prec_threshed = prec[rec >= t]

        if not len(cum_prec_threshed):
            precision = 0
        else:
            precision = np.max(cum_prec_threshed)

        ap += precision
    ap /= num_recall_points

    return ap


def compute_AP_integration(prec, rec):  # Post-2010 integration-based AP
    unique_rec, unique_rec_ind = np.unique(rec, return_index=True)

    interpolated_precision = np.maximum.accumulate(prec[unique_rec_ind][::-1])[::-1]
    recall_diff = np.append(np.diff(unique_rec), [0])

    ap = np.sum(interpolated_precision * recall_diff)

    return ap

def compute_mAP(all_preds, all_gts, labels, AP_type, matching_threshold):
    aps = []
    cum_precs = []
    cum_recs = []

    num_gts = np.zeros(shape=len(labels))
    for img_gts in all_gts:
        for gt in img_gts:
            num_gts[int(gt[0])] += 1

    for label_index in range(len(labels)):
        preds = np.array(all_preds[label_index])
        if not len(preds):
            # aps.append(0)  # class is ignored for mAP
            cum_precs.append([])
            cum_recs.append([])

            continue

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        preds_sorted = preds[np.argsort(-preds[:, 1])]

        gts_matched = [[] for _ in range(len(all_gts))]
        for i, pred in enumerate(preds_sorted):
            image_id = int(pred[0])
            gts = all_gts[image_id]

            label_mask = gts[:, 0] == label_index  # Because of entry per label vs entry per image  (could maybe use the same)
            gts = gts[label_mask]

            if not len(gts):
                fp[i] = 1

                continue

            ious = box_utils.calculate_iou(pred[-4:][None], gts[:, -4:])[0]
            gt_match_index = np.argmax(ious)
            gt_match_iou = ious[gt_match_index]

            if gt_match_iou < matching_threshold:
                fp[i] = 1
            else:
                if not len(gts_matched[image_id]):
                    tp[i] = 1
                    gts_matched[image_id] = np.zeros(shape=(len(gts)), dtype=bool)
                    gts_matched[image_id][gt_match_index] = True
                elif not gts_matched[image_id][gt_match_index]:
                    tp[i] = 1
                    gts_matched[image_id][gt_match_index] = True
                else:
                    fp[i] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        prec, rec = compute_prec_rec(cum_tp, cum_fp, num_gts[label_index])
        if AP_type == "sample":
            ap = compute_AP_sample(prec, rec)
        else:
            ap = compute_AP_integration(prec, rec)

        aps.append(ap)
        cum_precs.append(prec)
        cum_recs.append(rec)

    mAP = np.mean(aps)

    return mAP, aps, cum_precs, cum_recs


def evaluate(model, dataset, labels, AP_type="integrate", confidence_threshold=0.5):
    all_preds = [[] for _ in labels]  # Entry per label
    all_gts = []  # Entry per image

    for i, (image_path, gt_boxes, gt_labels) in enumerate(dataset):
        image, gt_boxes, gt_labels = prepare_testing(image_path, gt_boxes, gt_labels, model.input_shape)
        preds = model.inference(image, labels, confidence_threshold=0.01)  # Supposed to be 0.01 (as in SSD paper)

        # box_utils.plot_ious(gt_boxes, preds[1], image, labels=preds[0], confidences=preds[2])

        # preds = list(zip(*pred))
        # structured_preds = np.concatenate([np.full(len(pred[0]), i)[:, None], pred[2][:, None], pred[1]], axis=1)  # (image_id, confidence, cx, cy, w, h)
        for pred in zip(*preds):
            strucutred_pred = np.concatenate([[i, pred[2]], pred[1]])  # (image_id, confidence, cx, cy, w, h)
            label_index = labels.index(pred[0])
            all_preds[label_index].append(strucutred_pred)

        gts = np.concatenate([np.expand_dims(gt_labels, axis=1), gt_boxes], axis=-1)  # (label, cx, cy, w, h)  # np.array(gt_labels)[:, None]
        all_gts.append(gts)

    mAP, aps, precisions, recalls = compute_mAP(all_preds, all_gts, labels, AP_type, matching_threshold=confidence_threshold)
    plot_prec_rec(precisions, recalls, aps, labels)

    return mAP

"""
    from eval_imp import Evaluator
    e = Evaluator(model, len(labels), None)

    # e(300, 300, 8, None)
    p = list(zip(*all_preds_gts))
    flat = [(i, tr) for i, ew in enumerate(p[0]) for tr in ew]

    from box_utils import convert_to_coordinates, scale_box
    u = [[] for _ in range(len(labels))]
    for i, pred in flat:
        box = np.array(scale_box(convert_to_coordinates(pred[1][None])[0], (300, 300)), np.int16)
        u[labels.index(pred[0])].append([i, pred[2], box[0], box[1], box[2], box[3]])

    e.prediction_results = u

    asd = []
    for img in p[1]:
        asd.append([])
        for pred in img:
            box = np.array(scale_box(convert_to_coordinates(pred[1]), (300, 300)), np.int16)
            asd[-1].append([labels.index(pred[0]), box[0], box[1], box[2], box[3]])

    class DataGenerator:
        def __init__(self, labels, eval_neutral, image_ids):
            self.labels = labels
            self.eval_neutral = eval_neutral
            self.image_ids = image_ids
    e.data_generator = DataGenerator(asd, None, list(range(len(p[0]))))

    pre = e(300, 300, 8, return_precisions=True, return_recalls=True, return_average_precisions=True)
"""

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
