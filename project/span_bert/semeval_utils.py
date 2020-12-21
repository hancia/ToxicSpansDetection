def f1_semeval(pred_spans, true_spans):
    """
    F1 (a.k.a. DICE) operating on two lists of offsets (e.g., character).
    >>> assert f1([0, 1, 4, 5], [0, 1, 6]) == 0.5714285714285714
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(true_spans) == 0:
        return 1 if len(pred_spans) == 0 else 0
    nom = 2 * len(set(pred_spans).intersection(set(true_spans)))
    denom = len(set(pred_spans)) + len(set(true_spans))
    return nom / denom
