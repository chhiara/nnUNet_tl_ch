from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from nnunet.paths import preprocessing_output_dir


task_name = 'Task603__T1__AF_L__Apss'

# make sure to run the Task603 dataset conversion and
# nnUNet_plan_and_preprocess first!

#splits_fname = join(preprocessing_output_dir, task_name, 'splits_final.pkl')
#splits = load_pickle(splits_fname)

splits=[]

#comment out test data

images=[
    "1_AmCo__superset__AF_L", 
    #"2_BaAd__superset__AF_L", 
    "3_CaMi__superset__AF_L", 
    "4_CuMa__superset__AF_L",
    "5_DaMi__superset__AF_L", 
    "6_FeMa__superset__AF_L",
    "7_MaRo__superset__AF_L",
    "8_MasRob__superset__AF_L", 
    "9_MaLo__superset__AF_L",
    "10_MeGi__superset__AF_L",
    "11_MoLu__superset__AF_L",
    "12_PeFl__superset__AF_L",
    "13_PiAl__superset__AF_L",
    "14_SeDi__superset__AF_L",
    "15_ZaMa__superset__AF_L",
    #"16_ZoEl__superset__AF_L",
    "17_AmLo__superset__AF_L",
    "18_MazLor__superset__AF_L",
    #"19_IoPa__superset__AF_L",
    #"20_BeAl__superset__AF_L",
    "21_BrLu__superset__AF_L",
]

for i, image in enumerate(images):
    splits.append({})
    splits[i]["val"] = [image]
    training_list = list(filter(lambda im: im != image, images))
    splits[i]["train"] = training_list

# save the splits 
save_pickle(splits, join(preprocessing_output_dir, task_name, 'splits_final.pkl'))

# %%
