import torch


def get_err(target, impostors):
    thresholds, _ = torch.sort(torch.cat([target, impostors]))
    thresholds = torch.unique(thresholds)

    # Adding intermediate thresholds
    interm_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, interm_thresholds]))

    # Variable to store the min FRR, min FAR and their corresponding index
    min_index = 0
    final_FRR = 0
    final_FAR = 0

    for i, cur_thresh in enumerate(thresholds):
        pos_scores_threshold = target <= cur_thresh
        FRR = (pos_scores_threshold.sum(0)).float() / impostors.shape[0]
        del pos_scores_threshold

        neg_scores_threshold = target > cur_thresh
        FAR = (neg_scores_threshold.sum(0)).float() / impostors.shape[0]
        del neg_scores_threshold

        # Finding the threshold for EER
        if (FAR - FRR).abs().item() < abs(final_FAR - final_FRR) or i == 0:
            min_index = i
            final_FRR = FRR.item()
            final_FAR = FAR.item()

    # It is possible that eer != fpr != fnr. We return (FAR  + FRR) / 2 as EER.
    EER = (final_FAR + final_FRR) / 2

    return float(EER), float(thresholds[min_index])
