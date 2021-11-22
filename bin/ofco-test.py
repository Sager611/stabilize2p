import tifffile as tiff

import time
from ofco import motion_compensate
from ofco.utils import default_parameters

N = 100
path = '../data/200901_G23xU1/Fly1/Fly1/002_coronal/2p/denoised_red.tif'

key = range(N)

stack1 = tiff.imread(path, key=key)

param = default_parameters()
frames = [i for i in key]

t1 = time.perf_counter()
stack1_warped, _ = motion_compensate(stack1, None, frames, param)
t2 = time.perf_counter()
print(f'Elapsed {t2-t1:.4g}s | {(t2-t1)/N:.4g}s per frame')
