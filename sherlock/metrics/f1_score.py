# -*- coding: utf8 -*-
from sklearn.metrics import precision_recall_fscore_support


def compute_f1(preds, labels, exclude=[0]):
    label_set = sorted(list(set(labels + preds)))
    label_set.remove(exclude)
    prec, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro', labels=label_set)
    #n_gold = n_pred = n_correct = 0
    #for pred, label in zip(preds, labels):
    #    if pred != 0:
    #        n_pred += 1
    #    if label != 0:
    #        n_gold += 1
    #    if (pred != 0) and (label != 0) and (pred == label):
    #        n_correct += 1
    #if n_correct == 0:
    #    return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    #else:
    #    prec = n_correct * 1.0 / n_pred
    #    recall = n_correct * 1.0 / n_gold
    #    if prec + recall > 0:
    #        f1 = 2.0 * prec * recall / (prec + recall)
    #    else:
    #        f1 = 0.0
    return {"precision": prec, "recall": recall, "f1": f1}
