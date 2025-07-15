from torchvision import transforms
from datasets import getLoader
import torch
from model import CNNWithResidual
from utils import get_experiment_data, count_parameters


trainRoot = 'data/train'
testRoot = 'data/test'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNNWithResidual(input_channels=3, num_classes=100).to(device)
trainLoader, testLoader = getLoader(trainRoot, testRoot)

get_experiment_data(model, trainLoader, testLoader, device, "CNN with Residual Block")
