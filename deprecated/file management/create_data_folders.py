import os
import tkinter as tk
from tkinter import filedialog

# Get root directory from user
root = tk.Tk()
root.withdraw()
root = filedialog.askdirectory()
assert root, 'No root directory chosen'

freqs = [50 * x for x in range(2, 21)]
subdirs = ['CW', 'PRF0.10kHz', 'PRF1.50kHz']

for f in freqs:
    dirpath = '{}/{}kHz'.format(root, f)
    print(dirpath)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    for sd in subdirs:
        subdirpath = '{}/{}'.format(dirpath, sd)
        print('-->', subdirpath)
        if not os.path.exists(subdirpath):
            os.makedirs(subdirpath)

