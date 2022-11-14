import sys

import numpy as np
from PIL import Image
from pathlib import Path

if len(sys.argv) != 2:
    print(f'Usage: python3 {sys.argv[0]} path_to_data')
    sys.exit(-1)

mse_accumulate = []
for file_idx, fake_path in enumerate(Path(sys.argv[1]).glob('*fake*')):
    file_name = str(fake_path).split('/')[-1].split('_')[0]

    true_path = ''
    list_path = str(fake_path).split('/')
    for idx in range(len(list_path) - 1):
        true_path += ('/' + list_path[idx])
    true_path += ('/' + file_name + '_real_B.png')
    true_path = true_path[1:]

    if file_idx % 10 == 0:
        print(f'Processing file: {file_name}')

    image_fake = np.array(Image.open(str(fake_path))).astype('float')/255
    image_gt = np.array(Image.open(true_path)).astype('float')/255

    mse = ((image_gt - image_fake) ** 2).mean(axis=0)
    mse_accumulate.append(mse)

global_mse = np.array(mse_accumulate).mean()

print(f'The MSE for the test set is: {global_mse}')
