import os

from HE_Implementation import execute_func_HE
from AHE_Implementation import execute_func_AHE
from CLAHE_Implementation import execute_func_CLAHE

path = os.getcwd() + '/source pictures'

files = os.listdir(path)
files.sort()

for file in files:
    # execute_func_HE('source pictures/' + file, 'HE_rgb_target pictures/' + file)
    # execute_func_HE('source pictures/' + file, 'HE_hsv_target pictures/' + file)
    # execute_func_AHE('source pictures/' + file, 'AHE_rgb_target pictures/' + file)
    # execute_func_AHE('source pictures/' + file, 'AHE_hsv_target pictures/' + file)
    execute_func_CLAHE('source pictures/' + file, 'CLAHE_rgb_target pictures/' + file)
    # execute_func_CLAHE('source pictures/' + file, 'CLAHE_hsv_target pictures/' + file)
