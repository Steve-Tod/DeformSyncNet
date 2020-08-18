import numpy as np
from model.loss.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist

def calc_miou(tgt_shape, deform_shape, src_seg, tgt_seg, parts, nnd, return_CD=False):
    dist1, dist2, _, idx2 = nnd(deform_shape.transpose(1, 2).contiguous(), 
                                   tgt_shape.transpose(1, 2).contiguous())
    deform_seg = src_seg.gather(dim=1, index=idx2.long().cpu())
    miou = miou_shape(deform_seg.numpy(), tgt_seg.numpy(), parts)
    if return_CD:
        cd = dist1.mean() + dist2.mean()
        return miou, cd.item()
    else:
        return miou

def miou_shape(prediction, target, parts):
    # prediction : numpy array size n_elements
    # target : numpy array size n_elements
    # parts : list. e.g. [8.0, 9.0, 10.0, 11.0]

    ious = []

    for part in parts:
        # select a part
        tp = np.sum((prediction == int(part)) * (target == int(part)))  # true positive
        pc = np.sum((prediction == int(part)))  # predicted positive = TP + FP
        gt = np.sum((target == int(part)))  # GT positive
        if pc + gt - tp == 0:
            # print("union of true positive and predicted positive is empty")
            ious.append(1)
        else:
            ious.append(float(tp) / float(pc + gt - tp))  # add value

    return np.mean(ious)
