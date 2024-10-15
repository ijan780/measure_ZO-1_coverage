# MeasureZO-1Coverage.py - Calculates coverage of ZO-1 staining.

import os
from pathlib import Path
from parse import parse
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, exposure, util, filters, morphology
from skimage.measure import label, regionprops_table


def main():
    # Initialize final analysis spreadsheet columns
    samples = []
    treatments = []
    mucus = []
    stains = []
    sum_lengths = []

    # Import image(s) & initialize output path.
    pattern = '{stain}_{treatment}_{mucus}_{sample}.{filetype}'
    dirname = R'C:\Users\MVX\OneDrive - UW\Allbritton Lab\Hao\230918'   # set to image file directory

    # Initialize output path.
    outPath = Path(f'{dirname}_Analysis')
    os.makedirs(outPath, exist_ok=True)

    for filename in os.listdir(Path(dirname)):
        patternResult = parse(pattern, filename)
        treatments.append(patternResult['treatment'])
        mucus.append(patternResult['mucus'])
        stains.append(patternResult['stain'])
        samples.append(patternResult['sample'])
        dir = str(Path(dirname) / filename)
        sum_lengths.append(image_process(dir, filename[:-4], outPath, patternResult['stain']))

    # Output final data spreadsheet
    df_final = pd.DataFrame({"Treatment": treatments,
                             "Mucus": mucus,
                             "Stain": stains,
                             "Sample": samples,
                             "Sum_Length": sum_lengths})
    df_final.to_csv(outPath / f'Analysis.csv', header=True, index=False)

def image_process(dir, filename, outPath, stain):
    # Read 16 bit image, normalize, and convert to 8 bit.
    image = io.imread(dir)
    image_norm = exposure.rescale_intensity(image)
    image_norm = util.img_as_ubyte(image_norm)

    # Create binary mask.
    blur = filters.gaussian(image_norm, preserve_range=True)
    if stain == 'H':
        proc = 'Blur, Thresh (Li)'
        thresh_value = filters.threshold_li(blur)
    else:
        proc = 'Blur, Top Hat, Thresh(Otsu)'
        footprint = morphology.disk(4)
        blur = morphology.white_tophat(blur, footprint)
        thresh_value = filters.threshold_otsu(blur)
    thresh = blur > thresh_value

    # Measure properties from processed mask.
    labels = label(thresh)
    props = regionprops_table(labels, properties=('centroid',
                                                  'bbox',
                                                  'orientation',
                                                  'axis_major_length',
                                                  'axis_minor_length',
                                                  'num_pixels'))
    df_measure = pd.DataFrame(props)

    # Plot each step of image analysis process.
    fig, ax = plt.subplots(3, 1, figsize=(12, 6), sharey=True)

    ax[0].imshow(image_norm, cmap=plt.cm.gray)
    ax[0].set_title('Optimized B&C')

    ax[1].imshow(thresh, cmap=plt.cm.gray)
    ax[1].set_title(f'{proc}')

    ax[2].imshow(image, cmap=plt.cm.gray)
    ax[2].set_title('Centroids')

    for idx in range(0, df_measure.shape[0]):
        y0, x0 = df_measure['centroid-0'][idx], df_measure['centroid-1'][idx]
        orientation = df_measure['orientation'][idx]
        x1 = x0 + math.cos(orientation) * 0.5 * df_measure['axis_minor_length'][idx]
        y1 = y0 - math.sin(orientation) * 0.5 * df_measure['axis_minor_length'][idx]
        x2 = x0 - math.sin(orientation) * 0.5 * df_measure['axis_major_length'][idx]
        y2 = y0 - math.cos(orientation) * 0.5 * df_measure['axis_major_length'][idx]

        ax[2].plot((x0, x1), (y0, y1), '-r', linewidth=2.5)    # minor axis
        ax[2].plot((x0, x2), (y0, y2), '-r', linewidth=2.5)    # major axis
        ax[2].plot(x0, y0, '.g', markersize=15)

        minr, minc, maxr, maxc = df_measure['bbox-0'][idx], df_measure['bbox-1'][idx], df_measure['bbox-2'][idx], df_measure['bbox-3'][idx]
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax[2].plot(bx, by, '-b', linewidth=2.5)

    for a in ax.ravel():
        a.set_axis_off()

    fig.suptitle(filename, fontsize=14)
    fig.tight_layout()

    plt.savefig(outPath / f'{filename}_Analysis.png')
    plt.close()

    sum_length = xproject(df_measure)
    df_measure.to_csv(outPath / f'{filename}_Analysis.csv', header=True, index=False)
    return sum_length

# Calculates length of x-axis projections.
def xproject(df_measure):
    xprojleft = []
    xprojright = []
    for i in range(len(df_measure)):
        xprojlen = max(abs(df_measure.loc[i]['axis_minor_length'] * math.cos(df_measure.loc[i]['orientation'])),
                       abs(df_measure.loc[i]['axis_major_length'] * math.sin(df_measure.loc[i]['orientation'])))
        xprojleft.append(df_measure.loc[i]['centroid-1'] - 0.5 * xprojlen)
        xprojright.append(df_measure.loc[i]['centroid-1'] + 0.5 * xprojlen)
    df_measure[f'xprojleft'] = xprojleft
    df_measure[f'xprojright'] = xprojright

    overlaps = [None] * len(df_measure)
    for i in range(len(df_measure)):
        overlaps_single = {i}
        for j in range(len(df_measure)):
            first_int = [df_measure.loc[i]['xprojleft'],df_measure.loc[i]['xprojright']]
            sec_int = [df_measure.loc[j]['xprojleft'],df_measure.loc[j]['xprojright']]
            if max(first_int[0], sec_int[0]) < min(first_int[1], sec_int[1]):
                overlaps_single.add(j)
        overlaps[i] = overlaps_single
    # print(r'This is the list of overlaps: ' + str(overlaps))
    consolidated_overlaps = consolidate(overlaps)
    # print(r'This is the consolidated list of overlaps: ' + str(consolidated_overlaps))

    sum_length = 0
    for s in consolidated_overlaps:
        xprojleft_sub = []
        xprojright_sub = []
        for cell in s:
            xprojleft_sub.append(df_measure.loc[cell]['xprojleft'])
            xprojright_sub.append(df_measure.loc[cell]['xprojright'])
        sum_length += max(xprojright_sub) - min(xprojleft_sub)
        # print(f'Set {s} length: ' + str(max(xprojright_sub) - min(xprojleft_sub)) + ' = ' + str(max(xprojright_sub)) + ' - ' + str(min(xprojleft_sub)))
    # print(sum_length)
    return(sum_length)

# Consolidates overlapping segments.
def consolidate(sets):
    setlist = [s for s in sets if s]
    for i, s1 in enumerate(setlist):
        if s1:
            for s2 in setlist[i+1:]:
                intersection = s1.intersection(s2)
                if intersection:
                    s2.update(s1)
                    s1.clear()
                    s1 = s2
    return [s for s in setlist if s]

main()
