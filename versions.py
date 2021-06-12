import torch
import fastai
tv = torch.__version__
fav = fastai.__version__
cuda = torch.cuda.is_available()

print(f"Pytorch : {tv}, FastAI : {fav}, CUDA? : {cuda}")