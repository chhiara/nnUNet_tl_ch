from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from nnunet.paths import preprocessing_output_dir

tasks = [
    # "Task531__peaks__OR-left__BrainPTM",
    # "Task532__T1__IFOF_L__TractoInferno",
    # "Task533__T1__ILF_L__TractoInferno",
    #"Task534__T1__UF_L__TractoInferno",
    # "Task535__AP__IFOF_L__TractoInferno",
    # "Task536__AP__ILF_L__TractoInferno",
    #"Task537__AP__UF_L__TractoInferno",
    # "Task538__peaks__IFOF_L__TractoInferno",
    # "Task539__peaks__ILF_L__TractoInferno",
    #"Task540__peaks__UF_L__TractoInferno",
    # "Task541__T1__IFOF_L__Apss",
    # "Task542__T1__ILF_L__Apss",
    # "Task543__T1__UF_L__Apss",
    # "Task544__AP__IFOF_L__Apss",
    # "Task545__AP__ILF_L__Apss",
    # "Task546__AP__UF_L__Apss",
    # "Task547__peaks__IFOF_L__Apss",
    # "Task548__peaks__ILF_L__Apss",
    # "Task549__peaks__UF_L__Apss"
    "Task519__peaks__AF_L__Apss",
    "Task511__AP__AF_L__Apss",
    "Task512__AP__FAT_L__Apss",
    "Task520__peaks__FAT_L__Apss"
]

for task_name in tasks:
    print(task_name)
    #task_name = 'Task518__AP__PT_L__Apss'
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


