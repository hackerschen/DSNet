import torch

def cal_metric(output, target):
    iou, dice = iouAndDice(output, target)
    f1_score = F1_score(output, target)
    return iou, dice, f1_score

def iouAndDice(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection) / (union + smooth)
    dice = (2* iou) / (iou+1)
    return iou, dice

def F1_score(output, target):
    # two category
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    TP = ((output_ == 1) & (target_ == 1)).sum()
    TN = ((output_ == 0) & (target_ == 0)).sum()
    FN = ((output_ == 0) & (target_ == 1)).sum()
    FP = ((output_ == 1) & (target_ == 0)).sum()

    P = (TP + smooth) / ((TP + FP) + smooth)
    R = (TP + smooth) / ((TP + FN) + smooth)
    F1 = (2 * P * R + smooth) / (P + R + smooth)
    return F1

def iou_score(output, target):
    # because the number of channel is two, so mod the origin code, and this function only can be use in two-category
    smooth = 1e-5

    if torch.is_tensor(output):
        # output = torch.sigmoid(output).data.cpu().numpy()
        output = torch.argmax(output, 1)
    if torch.is_tensor(target):
        # target = target.data.cpu().numpy()
        target = target
    intersection = (output & target).sum()
    union = (output | target).sum()
    iou = (intersection + smooth) / (union + smooth)
    return iou

def miou_score(output, target, nclass):
    # the size of output is (N, C, H, W), so the shape of target is (N, 1, H, W)
    mini = 0
    maxi = nclass - 1
    nbins = nclass
    preds = torch.argmax(output, 1)

    preds = preds.float() * (target > 0).float()
    intersection = preds * (preds == target).float()

    area_inter = torch.histc(intersection, bins=nbins, min=mini, max=maxi)
    area_pred = torch.histc(preds, bins=nbins, min=mini, max=maxi)
    area_lab = torch.histc(target, bins=nbins, min=mini, max=maxi)
    area_union = area_lab + area_pred - area_inter

    miou = torch.mean(area_inter / area_union)
    return miou