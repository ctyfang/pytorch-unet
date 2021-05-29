from torchmetrics import Metric
import torch


class DiceCoefficient(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("intersection", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        self.intersection += torch.sum(preds == target)
        self.total += 2*preds.numel()

    def compute(self):
        return 2 * self.intersection.float() / self.total

