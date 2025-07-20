import torch
from torch import nn

class LinearRegressionModel(nn.Module):

	def __init__(self):
		super().__init__()
		self.device = "cuda" if torch.cuda.is_available() else "cpu" # Device agnostic
		self.layer = nn.Linear(in_features=1, out_features=1, device = self.device)

	def forward(self, x:torch.Tensor) -> torch.Tensor:
		return self.layer(x)