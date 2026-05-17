import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.metrics import AbsMetric
from LibMTL.loss import AbsLoss
from copy import deepcopy

criterion_mse = nn.MSELoss()

# Balance multi-task losses under EW (raw sum of task losses).
# PPI CE is usually much larger in magnitude (~5.x), so we down-scale it,
# while slightly up-scaling ECG/Anchor losses to avoid being dominated.
SHAPE_LOSS_SCALE = 3.0
PPI_LOSS_SCALE = 1.0
ANCHOR_LOSS_SCALE = 3.0

def normal_ecg_torch_01(ECG):
    for itr in range(ECG.size(dim=0)):
        ecg_min = torch.min(ECG[itr])
        ecg_max = torch.max(ECG[itr])
        denom = (ecg_max - ecg_min).clamp_min(1e-6)
        ECG[itr] = (ECG[itr] - ecg_min) / denom
    return ECG

def cross_entropy_loss_shape(ecg_rcon, ecg_gts):
    loss = nn.CrossEntropyLoss()
    ecg_rcon = ecg_rcon.squeeze(1)
    possi = ecg_gts.squeeze(1).softmax(dim=1)
    return loss(ecg_rcon, possi)

def cross_entropy_loss_ppi(ecg_rcon, ecg_gts):
    # Convert padded waveform labels to class indices (cycle length bins).
    # This keeps CE numerically meaningful and aligned with argmax-based PPI error.
    ecg_rcon, ecg_gts = ecg_rcon.squeeze(1), ecg_gts.squeeze(1)
    counts = ecg_gts.size(1) - (ecg_gts == -10).sum(dim=1)
    targets = (counts - 1).long().clamp(min=0, max=ecg_rcon.size(1) - 1)
    return F.cross_entropy(ecg_rcon, targets)

def ppi_error(ecg_rcon, ecg_gts):
    ecg_rcon, ecg_gts = ecg_rcon.squeeze(1), ecg_gts.squeeze(1)
    counts = ecg_gts.size(1)-(ecg_gts == -10).sum(dim=1)+1  # ppi_gts
    batch_indices = torch.arange(ecg_gts.size(0))
    ppi_pred = ecg_rcon.argmax(dim=1)
    return torch.mean(torch.abs(ppi_pred - counts)/200)

# def anchor_error(ecg_rcon, ecg_gts):
#     ecg_rcon, ecg_gts = ecg_rcon.squeeze(1), ecg_gts.squeeze(1)

def r2_score(y_true, y_pred):
    y_true_mean = torch.mean(y_true)
    ss_total = torch.sum((y_true - y_true_mean) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_residual / ss_total
    return r2

# ECG shape Metric
class shapeMetric(AbsMetric):
    def __init__(self):
        super(shapeMetric, self).__init__()
        self.mse_record = []
        self.ce_record = []
        self.norm_mse_record = []
    def update_fun(self, pred, gt):
        gt = torch.clone(gt).detach()
        gt = normal_ecg_torch_01(gt).to(pred.device)
        mse = criterion_mse(pred, gt)
        pred_norm = normal_ecg_torch_01(torch.clone(pred).detach())
        self.mse_record.append(mse.item())
        self.norm_mse_record.append(criterion_mse(pred_norm, gt).item())
        self.bs.append(pred.size()[0])
        ce = cross_entropy_loss_shape(pred, gt)
        self.ce_record.append(ce.item())
    def score_fun(self):
        records = np.array(self.mse_record)
        batch_size = np.array(self.bs)
        mse = (records*batch_size).sum()/(sum(batch_size))
        records = np.array(self.ce_record)
        ce = (records*batch_size).sum()/(sum(batch_size))
        norm_mse = (np.array(self.norm_mse_record)*batch_size).sum()/(sum(batch_size))
        return [norm_mse, mse, ce]
    def reinit(self):
        self.mse_record = []
        self.ce_record = []
        self.norm_mse_record = []
        self.bs = []
# mse loss as shapeloss
class shapeLoss(AbsLoss):
    def __init__(self):
        super(shapeLoss, self).__init__()
    def compute_loss(self, pred, gt):
        gt = torch.clone(gt).detach()
        gt = normal_ecg_torch_01(gt).to(pred.device)
        return SHAPE_LOSS_SCALE * criterion_mse(pred, gt)
        # return r2_score(pred, gt)
# PPI Metric
class ppiMetric(AbsMetric):
    def __init__(self):
        super(ppiMetric, self).__init__()
        self.ce_record = []
        self.ppi_record = [] # error in seconds
    def update_fun(self, pred, gt):
        ce = cross_entropy_loss_ppi(pred, gt)
        self.ce_record.append(ce.item())
        self.bs.append(pred.size()[0])
        ppi = ppi_error(pred, gt)
        self.ppi_record.append(ppi.item())
    def score_fun(self):
        records = np.array(self.ce_record)
        batch_size = np.array(self.bs)
        ce = (records*batch_size).sum()/(sum(batch_size))
        records = np.array(self.ppi_record)
        ppi = (records*batch_size).sum()/(sum(batch_size))
        return [ppi, ce]
    def reinit(self):
        self.ce_record = []
        self.ppi_record = []
        self.bs = []

# ce loss for ppi
class ppiLoss(AbsLoss):
    def __init__(self):
        super(ppiLoss, self).__init__()
    def compute_loss(self, pred, gt):
        return PPI_LOSS_SCALE * cross_entropy_loss_ppi(pred, gt)
    
# anchor Metric only use mse
class anchorMetric(AbsMetric):
    def __init__(self):
        super(anchorMetric, self).__init__()
    def update_fun(self, pred, gt):
        gt = torch.clone(gt).detach()
        gt = normal_ecg_torch_01(gt).to(pred.device)
        pred_norm = normal_ecg_torch_01(torch.clone(pred).detach())
        mse = criterion_mse(pred_norm, gt)
        self.record.append(mse.item())
        self.bs.append(pred.size()[0])
    def score_fun(self):
        records = np.array(self.record)
        batch_size = np.array(self.bs)
        return [(records*batch_size).sum()/(sum(batch_size))]
class anchorLoss(AbsLoss):
    def __init__(self):
        super(anchorLoss, self).__init__()
        self.sigma = 3
        self.gaussian_kernel = None
    
    def _create_gaussian_kernel(self, size, device):
        if self.gaussian_kernel is None or self.gaussian_kernel.device != device or self.gaussian_kernel.shape[0] != size:
            x = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
            kernel = torch.exp(-(x ** 2) / (2 * self.sigma ** 2))
            kernel = kernel / kernel.sum()
            self.gaussian_kernel = kernel
        return self.gaussian_kernel
    
    def compute_loss(self, pred, gt):
        gt = torch.clone(gt).detach()
        gt = normal_ecg_torch_01(gt).to(pred.device)

        # Accept both [B, L] and [B, 1, L] label layouts.
        if gt.dim() == 2:
            gt_1d = gt.unsqueeze(1)
        elif gt.dim() == 3 and gt.size(1) == 1:
            gt_1d = gt
        elif gt.dim() == 4 and gt.size(1) == 1:
            gt_1d = gt.squeeze(1)
        else:
            raise ValueError(f"Unexpected gt shape for anchorLoss: {tuple(gt.shape)}")
        
        kernel = self._create_gaussian_kernel(gt_1d.shape[-1], gt_1d.device)
        gt_smooth = torch.nn.functional.conv1d(
            gt_1d,
            kernel.unsqueeze(0).unsqueeze(0),
            padding=gt_1d.shape[-1] - 1
        )[:, 0, :gt_1d.shape[-1]]

        if pred.dim() == 3 and pred.size(1) == 1:
            gt_smooth = gt_smooth.unsqueeze(1)
        
        return ANCHOR_LOSS_SCALE * criterion_mse(pred, gt_smooth)