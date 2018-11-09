#!/usr/bin/python3
import sys
import os
import imageio
from imgaug import augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


print(sys.argv[1])
target_dir = 'augmented_images/'+sys.argv[1]+'/'
ensure_dir(target_dir)

total_count = 0
for arg in sys.argv[2:]:
    file_name = arg.split('/')[2]
    only_name = file_name.split('.')[0]
    only_type = file_name.split('.')[1]
    image = imageio.imread(arg)

    blurs = np.arange(0.1, 0.9, 0.2)
    noises = np.arange(0.01, 0.07, 0.02)
    contrast_directions = np.arange(0.4, 0.6, 0.1)
    contrast_strength = np.arange(1.0, 1.7, 0.2)
    brightness_intensities = np.arange(30, 61, 10)
    count = 0
    results_grid = []
    results_row = []
    columns = 30
    for blur in blurs:
        for noise in noises:
            for direction in contrast_directions:
                for strength in contrast_strength:
                    for brightness in brightness_intensities:
                        count += 1
                        augmented = iaa.Sequential([
                            iaa.GaussianBlur(blur),
                            iaa.AdditiveGaussianNoise(scale=(0.0, noise*255), per_channel=0.5),
                            iaa.ContrastNormalization((direction, strength)),
                            iaa.Add((brightness*-1, brightness))
                        ]).augment_image(image)
                        results_row.append(augmented)
                        imageio.imwrite(target_dir+only_name+'_augmented_'+str(count)+'.'+only_type, augmented)
                        if count % columns == 0:
                            results_grid.append(np.concatenate(results_row, axis=1))
                            results_row = []
    total_count += count
    print("\t%d new images generated" % count)

    # plt.subplot2grid((1, 2), (0, 0)), plt.title('%s' % file_name), plt.imshow(image)
    # plt.subplot2grid((1, 2), (0, 1)), plt.title('results for %s' % file_name), plt.imshow(np.concatenate(results_grid, axis=0))
    # plt.show()

print("%d total images generated" % total_count)
