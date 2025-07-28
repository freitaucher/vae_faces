import cv2
import numpy as np
#
from utils_model import VAE, vae_loss, collate_to_cpu
#
import torch
from torch.utils.data import DataLoader, default_collate, dataset
from torchvision import datasets, transforms
#
import json
#
import os
import shutil


# Center Crop and Image Scaling functions in OpenCV
# Date: 11/10/2020
# Written by Nandan M

import cv2


def scale_image(img, factor=1):
    """Returns resize image by scale factor.
    This helps to retain resolution ratio while resizing.

    Args:
    img: image to be scaled
    factor: scale factor to resize
    """
    return cv2.resize(img, (int(img.shape[1]*factor), int(img.shape[0]*factor)))


def center_crop(img, dim):
    """Returns center cropped image

    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped
    """
    width, height = img.shape[1], img.shape[0]

    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2)
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img


def handle(model, x, device='cuda'):
    with torch.no_grad():
        z = model.encoder(x.unsqueeze(0))  # .squeeze(0)
        mu, logvar = z.chunk(2, dim=1)
        z = model.reparameterize(mu, logvar)
        recon = (model.decoder(z)).squeeze(0).cpu().numpy()
        recon = np.swapaxes(recon, 0, -1)
        recon = (recon * 255).astype(np.uint8)
        recon = np.swapaxes(recon, 0, 1)
        recon = cv2.cvtColor(recon, cv2.COLOR_BGR2RGB)
    return recon


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


# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert BGR to RGB
    # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = scale_image(frame, factor=0.3)
    frame = center_crop(frame, (128, 128))
    frame_tensor = torch.from_numpy((np.swapaxes(frame, 0, -1)/255).astype(np.float32)).to(device)
    img = handle(model, frame_tensor, device=config['device'])
    # print(img.shape)
    # quit()

    # Display FPS (performance monitor)

    # cv2.putText(frame, f"FPS: {int(cap.get(cv2.CAP_PROP_FPS))}", (10, 30),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    combined = cv2.hconcat([frame, img])
    cv2.imshow('check', combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
