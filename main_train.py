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
        model = torch.load(outdir + '/last.pth')
    # """
    if os.path.exists(outdir + "/best.txt"):
        f = open(outdir + "/best.txt", 'r')
        losses = f.read()
        losses = [float(l) for l in losses.splitlines()]
        f.close()
        loss_min = losses[-1]

    if os.path.exists(outdir + "/best_train.txt"):
        f = open(outdir + "/best_train.txt", 'r')
        losses_train = f.read()
        losses_train = [float(l) for l in losses_train.splitlines()]
        f.close()
        loss_min_train = losses_train[-1]
    # """
    restarted = True

else:
    shutil.rmtree(outdir, ignore_errors=True)
    os.mkdir(outdir)


if eval(config["epochs_count_reset"]):
    epoch = 0

x_worst, y_worst = None, None
index_worst = 0
loss_worst = 0
# criterion_mse = nn.MSELoss()
# criterion_vae = vae_loss()

time_tot_s = time()


# flist = glob.glob(config["path_data"]+'/*.jpg')
# flist_train = flist[:int(0.75*len(flist))]
# flist_val = flist[int(0.75*len(flist)):]

transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
])


train_data = datasets.CelebA(root=config["path_data"], split='train', transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=config["batch_size_train"], shuffle=True, collate_fn=collate_to_cpu, pin_memory=True, num_workers=16)
val_data = datasets.CelebA(root=config["path_data"], split='valid', transform=transform, download=True)
val_loader = DataLoader(val_data, batch_size=config["batch_size_val"], shuffle=True, collate_fn=collate_to_cpu, pin_memory=True, num_workers=16)

# print(len(flist), len(flist_train), len(flist_val))
# quit()

while epoch < config["epochs"]:

    time_s = time()

    # x = get_batch(config["batch_size"], flist_train)
    # x = np.swapaxes(x, 1, -1)  # .astype(np.float32)
    # x = torch.tensor(x).to(device)

    x, _ = next(iter(train_loader))
    x = x.to(device)

    model.train()

    opt.zero_grad()

    if eval(config["repeat_worst"]) and epoch > 0 and not restarted:
        x[index_worst] = x_worst
        for i in range(config["batch_size_train"]):
            seed = random.uniform(0, 1)
            if seed < config["admixture_of_worse"]:
                x[i] = x_worst

    restarted = False  # already passed

    x_out, mue, logvar = model(x)

    x_out = x_out[:, :, :x.shape[2], : x.shape[3]]
    # print(x_out.shape, x.shape)
    assert x_out.shape == x.shape, "Output and target shapes must match"
    loss = vae_loss(x_out, x, mue, logvar)
    loss_aux = ((x - x_out)**2).sum(dim=(-3, -2, -1))
    index_worst = int(torch.argmax(loss_aux))
    loss_worst = loss_aux[index_worst].item()
    x_worst = copy.deepcopy(x[index_worst])  # keep the worst for repeatition
    loss.backward()
    opt.step()
    loss_train = loss.item()

    model.eval()

    with torch.no_grad():  # , autocast(config['device'])

        xv, _ = next(iter(val_loader))
        xv = xv.to(device)
        xv_out, mue, logvar = model(xv)

        assert xv_out.shape == xv.shape, "Output and target shapes must match"

        try:
            loss_val = vae_loss(xv_out, xv, mue, logvar)
        except:
            print(xv.shape, xv_out.shape)

        loss_val = loss_val.item()

        if loss_val < loss_min and epoch > config["epochs_warmup"]:
            loss_min = copy.deepcopy(loss_val)
            checkpoint = {'epoch': epoch,  'model': model.state_dict(), 'optimizer': opt.state_dict()}
            torch.save(checkpoint, outdir + '/best.pth')
            with open(outdir + "/best.txt", "a") as f:
                f.write(str(loss_min) + '\n')
                f.close()
            # os.system('send2telegram  "new loss_min: %8.5f"' % loss_min)

        if loss_train < loss_train_min and epoch > config["epochs_warmup"]:
            loss_train_min = copy.deepcopy(loss_train)
            checkpoint = {'epoch': epoch,  'model': model.state_dict(), 'optimizer': opt.state_dict()}
            torch.save(checkpoint, outdir + '/best_train.pth')
            with open(outdir + "/best_train.txt", "a") as f:
                f.write(str(loss_train_min) + '\n')
                f.close()

    time_f = time()

    if epoch % 10 == 0:
        checkpoint = {'epoch': epoch,  'model': model.state_dict(), 'optimizer': opt.state_dict()}
        torch.save(checkpoint, outdir + '/last.pth')

        xv = xv.detach().cpu().numpy()
        xv_out = xv_out.detach().cpu().numpy()
        xv = np.swapaxes(xv, 1, -1)
        xv = np.swapaxes(xv, 1, 2)*255
        xv_out = np.swapaxes(xv_out, 1, -1)
        xv_out = np.swapaxes(xv_out, 1, 2)*255

        for i in range(config["batch_size_val"]):
            outfname = scratch + '/' + str(epoch).zfill(7) + '.png'
            # cv2.imwrite(outfname,  cv2.cvtColor(cv2.vconcat([cv2.hconcat([xv[i], xv_out[i]]) for i in range(config["batch_size_val"])]), cv2.COLOR_BGR2RGB))
            cv2.imwrite(outfname,  cv2.cvtColor(cv2.hconcat([cv2.vconcat([xv[i], xv_out[i]]) for i in range(config["batch_size_val"])]), cv2.COLOR_BGR2RGB))
            # cv2.imwrite(outfname,  cv2.vconcat([cv2.hconcat([xv[i], xv_out[i]]) for i in range(min(10, config["batch_size_val"]))]))

        scratch_files = sorted(glob.glob(scratch+'/*.png'))
        if len(scratch_files) > config["scratch_len"]:
            [os.unlink(f) for f in scratch_files[:-config["scratch_len"]]]

    print('epoch %10d  loss_t  %8.5f  loss_v  %8.5f  loss_min  %8.5f  loss_t_min  %8.5f ' % (epoch, loss_train, loss_val, loss_min, loss_train_min), 'index_worst %5d' % index_worst, 'loss_worst: %10.5f' % loss_worst, 'time: %8.4f' % (time_f-time_s))

    epoch += 1

    f = open("config.json")
    config = json.load(f)
    f.close()

# time_tot_f = time()
# print('time for 1000 epochs: %8.5f' % (time_tot_f-time_tot_s), 'per epoch: %8.5f' % ((time_tot_f-time_tot_s)/1000))
