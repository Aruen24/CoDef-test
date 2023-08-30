import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

root_dir = '/home/xxx/code/CoDeF/all_sequences'
name = 'beauty_1'

msk_folder = f'{root_dir}/{name}/{name}_masks'
img_folder = f'{root_dir}/{name}/{name}'
frg_folder = f'{root_dir}/{name}/{name}_masks_0'
bkg_folder = f'{root_dir}/{name}/{name}_masks_1'
os.makedirs(frg_folder, exist_ok=True)
os.makedirs(bkg_folder, exist_ok=True)

files = glob(msk_folder + '/*.png')
num = len(files)

for i in tqdm(range(num)):
    file_n = os.path.basename(files[i])
    mask = cv2.imread(os.path.join(msk_folder, file_n), 0)
    mask[mask > 0] = 1
    frg = cv2.imread(os.path.join(img_folder,
                                    file_n[:-4] + ".png")) * mask[:, :, None]
    bkg = cv2.imread(os.path.join(
        img_folder, file_n[:-4] + ".png")) * (1 - mask[:, :, None])
    cv2.imwrite(os.path.join(msk_folder, file_n), mask * 255)
    cv2.imwrite(os.path.join(frg_folder, file_n), frg)
    cv2.imwrite(os.path.join(bkg_folder, file_n), bkg)
