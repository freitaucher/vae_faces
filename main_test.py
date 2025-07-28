import json
import os
import glob
import shutil
import torch
import numpy as np
from utils_model import VAE, vae_loss, collate_to_cpu
import copy
from time import time
import random
import torch.nn as nn
import cv2
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, default_collate, dataset
from torchvision import datasets, transforms


def interpolate(model, x1, x2, n=10, tag='ae', device='cuda'):
    """
  Interpolate between two input images and display the sequence using OpenCV.
   Args:
        model: Trained AE/WAE model.
        x1, x2: Input images(PyTorch tensors, shape[1, 784]).
        n: Number of interpolation steps.
    """

    # aaa = x1.view(1, -1)
    # print(aaa.shape)
    # quit()

    with torch.no_grad():
        # Encode to latent space
        # z1 = model.encoder(x1.view(1, -1))
        # z2 = model.encoder(x2.view(1, -1))
        z1 = model.encoder(x1.unsqueeze(0)).squeeze(0)
        z2 = model.encoder(x2.unsqueeze(0)).squeeze(0)
        # print(z1.shape, z2.shape)

        # Linear interpolation in latent space
        alphas = torch.linspace(0, 1, n)
        interpolations = torch.stack([(1 - a) * z1 + a * z2 for a in alphas])
        # print(interpolations.shape)
        mu, logvar = interpolations.chunk(2, dim=1)
        z = model.reparameterize(mu, logvar)
        # print(z.shape)
        # quit()

        # Decode interpolations
        recon = model.decoder(z).cpu().numpy()
        # recon = recon.reshape(-1, 128, 128, 3)  # Reshape to (n, 28, 28)
        recon = np.swapaxes(recon, 1, -1)
        # print(recon.shape)
        # quit()
        # Normalize to [0, 255] and convert to uint8
        recon = (recon * 255).astype(np.uint8)
        # Concatenate images horizontally
        interpolation_strip = cv2.hconcat([np.swapaxes(img, 0, 1) for img in recon])
        interpolation_strip = cv2.cvtColor(interpolation_strip, cv2.COLOR_BGR2RGB)
        return interpolation_strip


def get_batch_aux(flist, k):
    img = cv2.imread(flist[k])
    dw, dh = img.shape[1]-128, img.shape[0]-128
    h0 = random.randint(0, dh)
    w0 = random.randint(0, dw)
    return img[h0:h0+128, w0:w0+128, :]


def get_batch(batch_size, flist):
    index = np.random.choice(len(flist), batch_size, replace=False)
    # time_s = time()
    x = Parallel(n_jobs=32)(delayed(get_batch_aux)(flist, index[ii]) for ii, _ in enumerate(index))
    # time_f = time()
    # print('read time pro batch %6.2f' % (time_f-time_s))
    x = np.array(x).astype(np.float32)/255

    """
    for i in range(batch_size):
        # if random.choice([True, False]):
        #    x[i] = x[i].T
        if random.choice([True, False]):
            x[i] = np.flip(x[i])
        if random.choice([True, False]):
            x[i] = np.flipud(x[i])
        if random.choice([True, False]):
            x[i] = np.fliplr(x[i])
    """

    return x


f = open("config.json")
config = json.load(f)
f.close()

device = torch.device(config["device"])

model = VAE(latent_dim=128)
model.to(device)

loss_min, loss_val, loss_train_min, loss_train = 9999999, 0, 9999999, 0

scratch = config["path_scratch_train"]
shutil.rmtree(scratch, ignore_errors=True)
os.mkdir(scratch)

epoch = 0
opt = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-8)

restarted = False
outdir = config["path_save_model"]
if os.path.exists(outdir + "/last.pth"):
    try:
        print("\n...load last train state\n")
        checkpoint = torch.load(outdir + '/last.pth')
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint['epoch']
    except:
        print('\n weights are not state_dict, try to load directly..')
        model = torch.load(outdir + '/best.pth')


transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
])

test_data = datasets.CelebA(root=config['path_data'], split='test', transform=transform, download=True)
test_loader = DataLoader(test_data, batch_size=config["batch_size_test"], shuffle=True, collate_fn=collate_to_cpu, pin_memory=True, num_workers=16)


model.eval()

with torch.no_grad():  # , autocast(config['device'])

    img_seq = []
    xv, _ = next(iter(test_loader))
    xv = xv.to(device)
    for i in range(config["batch_size_test"]//2):
        img_seq.append(interpolate(model, xv[i], xv[i+1], n=8, tag='ae', device=device))
    cv2.imwrite('interpolate_vae.png', cv2.vconcat(img_seq))
