import torch
import torch.nn as nn
import torch.nn.functional as F



class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        probs = torch.sigmoid(logits)

        pt = torch.where(targets==1.0, probs, 1.0 - probs)

        alpha_weight = torch.weight(targets==1.0, self.alpha, 1.0 - self.alpha)

        focal_loss = alpha_weight * torch.pow(1.0 - pt, self.gamma) * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            print("Wrong function was given to loss function")



    
