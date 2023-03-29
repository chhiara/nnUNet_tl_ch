from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from nnunet.paths import preprocessing_output_dir


task_name = 'Task530__peaks__CST-left__BrainPTM'
# make sure to run the Task603 dataset conversion and
# nnUNet_plan_and_preprocess first!

splits_fname = join(preprocessing_output_dir, task_name, 'splits_final.pkl')
splits = load_pickle(splits_fname)

nr_splits = 3
new_splits=[]

if(len(splits) == 5):
    all_images = np.concatenate((splits[0]["val"],splits[0]["train"])).tolist()
    np.random.shuffle(all_images)
    images_array_splits = np.array_split(all_images, nr_splits)

    for index in range(nr_splits):
        new_splits.append({})

        if(index==0):
            new_splits[index]["train"] = np.concatenate((images_array_splits[1],images_array_splits[2])).tolist()
        elif(index==1):
            new_splits[index]["train"] = np.concatenate((images_array_splits[0],images_array_splits[2])).tolist()
        else:
            new_splits[index]["train"] = np.concatenate((images_array_splits[0],images_array_splits[1])).tolist()

        new_splits[index]["val"] = images_array_splits[index].tolist()
    # save the splits 
    save_pickle(new_splits, join(preprocessing_output_dir, task_name, 'splits_final.pkl'))


