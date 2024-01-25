import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchtext.vocab import vocab
import torchtext.transforms as T
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import math
import janome
from janome.tokenizer import Tokenizer
import spacy
from collections import Counter

df = pd.read_excel("./JEC_basic_sentence_v1-3.xlsx", header=None)
