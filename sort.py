#!/usr//bin/env python3
from __future__ import absolute_import, division, print_function

import curses
import io
import matplotlib as mpl
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
import multiprocessing as mp
import numpy as np
import os
from os import listdir
from os.path import isfile, isdir, join
from PIL import Image
import re
from scipy.io import loadmat
from sklearn.preprocessing import normalize
from skimage.transform import rescale, resize
import sys
from threading import Timer
import time

def collect_images(path):
    """ Import all images from path recursively. Supports mat files or jpg files

    Parameters:
    path (str): path to dataset

    Returns:
    List: list of full file paths from current directory
    """
    contents = listdir(path)
    paths = []
    for c in sorted(contents):
        new_path = join(path, c)

        if isfile(new_path) and new_path.endswith("frontal.jpg") or new_path.endswith(".jpeg") or new_path.endswith(".mat"):
            # Check age of patient, only use ages 0 to 5 inclusive
            regexp = re.compile(r'0[0-5]ys')
            if not regexp.search(new_path):
                continue
            paths.append(new_path)
        elif isdir(new_path):
            # If new_path is a directory, recursively search it for samples
            new_paths = collect_images(new_path)
            if len(new_paths) > 0:
                paths += new_paths
    return paths



def sort_image(image_path, line, line_path, noline_path):
    """Writes image_path to line_path of noline_path depending on boolean value of line"""
    filename = ""
    if image_path.endswith("jpg") or image_path.endswith("jpeg"):
        filename = image_path[image_path.index("patient"):] #All file names are the same format
    elif image_path.endswith("mat"):
        filename = image_path[image_path.index("cropped"):][8:] # All file names are the same format

    path = ""
    if line:
        path = join(line_path, filename)
        # Write filename to file
        with open("line.txt", "a") as myfile:
            myfile.write(filename + "\n")
    else:
        path = join(noline_path, filename)
        # Write filename to file
        with open("noline.txt", "a") as myfile:
            myfile.write(filename + "\n")
    return img, path

def write_image(data):
    """Save image in data[0] to file at the path in data[1]"""
    img = data[0]
    path = data[1]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, "PNG")

def load_image(image_path):
    """Load image from file, depending on the file type"""
    image = None
    if image_path.endswith("jpg") or image_path.endswith("jpeg"):
        image = Image.open(image_path)
        pixels = list(image.getdata())
        width, height = image.size
        pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
        size = image.size
        image = np.asarray(pixels)
        image = image * (10**19)
    elif image_path.endswith("mat"):
        image = loadmat(image_path, appendmat=False)['dxImage']['img'][0][0]
    image_rescaled = image
    return image_rescaled


def display_image(image_path, f, axarr):
    """Update the image data in figure f"""
    image = load_image(image_path)
    if image_path.endswith("jpg") or image_path.endswith("jpeg"):
        axarr.set_data(image)
        f.canvas.flush_events()
    elif image_path.endswith("mat"):
        max_value = np.max(image)
        min_value = np.min(image)
        full_range = max_value - min_value
        axarr[0,0].set_data(image)
        axarr[0,1].set_data(image)
        axarr[1,0].set_data(image)
        axarr[1,1].set_data(image)
        axarr[0,0].set_clim(vmin=min_value, vmax=max_value)
        axarr[0,1].set_clim(vmin=min_value, vmax=min_value + (0.75 * full_range))
        axarr[1,0].set_clim(vmin=min_value + (0.25 * full_range), vmax=min_value + (0.75 * full_range))
        axarr[1,1].set_clim(vmin=min_value + (0.25 * full_range), vmax=max_value)
        f.canvas.flush_events()

    plt.pause(0.001)

def print_win_message(win):
    win.clear()
    win.addstr("Press <- for line, -> for no line, and [enter] to exit:")

def setup_win(win):
    win.nodelay(True)
    key=""
    print_win_message(win)

def main(win):
    if len(sys.argv) < 4:
        print("Please specify a data directory, a line directory, and a no line directory\n")
        sys.exit()
    path = str(sys.argv[1])
    line_path = sys.argv[2]
    noline_path = sys.argv[3]

    # Collect images
    images = sorted(collect_images(path))

    pool = mp.Pool(processes=4)             # Start a worker processes.
    setup_win(win)

    image_iter = 0
    if len(sys.argv) > 4:
        image_iter = int(sys.argv[4])
    start_image = image_iter

    # Initialize plot
    plt.ion()
    f = None
    axarr = None
    if images[0].endswith("jpg"):
        image = load_image(images[0])
        f = plt.figure(figsize=(10,10))
        ax = f.add_subplot()
        im = ax.imshow(load_image(images[0]), cmap="gray")
        axarr = im
        ax.axis('off')
        plt.tight_layout()
        plt.pause(0.001)
    else:
        image = load_image(images[0])
        f, ax = plt.subplots(2,2, num=10, clear=True, figsize=(10, 10))
        max_value = np.max(image)
        min_value = np.min(image)
        full_range = max_value - min_value
        axarr = np.array([[None,None], [None,None]])
        axarr[0,0] = ax[0,0].imshow(image, cmap="gray")
        axarr[0,1] = ax[0,1].imshow(image, cmap="gray", vmin=min_value, vmax=min_value + (0.75 * full_range))
        axarr[1,0] = ax[1,0].imshow(image, cmap="gray", vmin=min_value + (0.25 * full_range), vmax=min_value + (0.75 * full_range))
        axarr[1,1] = ax[1,1].imshow(image, cmap="gray", vmin=min_value + (0.5 * full_range), vmax=max_value)
        ax[0,0].axis('off')
        ax[0,1].axis('off')
        ax[1,0].axis('off')
        ax[1,1].axis('off')
        plt.tight_layout()
        plt.pause(0.001)

    # Display first image
    image_plt = display_image(images[image_iter], f, axarr)
    print_win_message(win)
    while image_iter < len(images):
        try:
            key = win.getkey()
            if key == os.linesep:
                break
            elif str(key) == "KEY_LEFT":
                win.addstr(" Line selected for image %d" % image_iter)
                win.refresh()
                # Save image path to file asynchronously
                pool.apply_async(sort_image, [images[image_iter], True, line_path, noline_path])
                image_iter+=1
                display_image(images[image_iter], f, axarr)
                print_win_message(win)
                win.refresh()
            elif str(key) == "KEY_RIGHT":
                win.addstr(" No line selected for image %d" % image_iter)
                win.refresh()
                # Save image path to file asynchronously
                pool.apply_async(sort_image, [images[image_iter], False, line_path, noline_path])
                image_iter+=1
                display_image(images[image_iter], f, axarr)
                print_win_message(win)
                win.refresh()
        except Exception as e:
            # No input
            pass
        time.sleep(0.01)
    return start_image, image_iter
    pool.close()
    pool.join()


def test():
    """Test the code without wrapping it in curses"""
    if len(sys.argv) < 4:
        print("Please specify a data directory, a line directory, and a no line directory\n")
        sys.exit()
    path = str(sys.argv[1])
    line_path = sys.argv[2]
    noline_path = sys.argv[3]

    images = sorted(collect_images(path))
    img, path = sort_image(images[0], True, line_path, noline_path)
    plt.ion()
    f = plt.figure()
    ax = f.add_subplot()
    im = ax.imshow(load_image(images[0]), cmap="gray")
    axarr = im
    plt.pause(0.001)
    #f, axarr = plt.subplots(2,2, num=10, clear=True, figsize=(12, 10))
    print(f)
    time.sleep(1)
    display_image(images[1], f, axarr)
    time.sleep(1)
    #write_image((img, path))


#test()
initial_count, sorted_count = curses.wrapper(main)
print("Sorted %d images, resume on image %d" % (sorted_count-initial_count, sorted_count))


# TODO: Time labeling and calculate statistics and remaining time estimates

with open("resume.txt", "w") as myfile:
    myfile.write("{} {} {} {} {}\n".format(sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3], sorted_count))
