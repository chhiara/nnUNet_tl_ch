from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from nnunet.paths import preprocessing_output_dir

tasks = [
    # "Task707__T1__ILF_L__TractoInferno",
    # "Task708__AP__ILF_L__TractoInferno",
    # "Task709__peaks__ILF_L__TractoInferno",
    # "Task704__T1__IFOF_L__TractoInferno",
    # "Task705__AP__IFOF_L__TractoInferno",
    # "Task706__peaks__IFOF_L__TractoInferno",
    # "Task740__T1__ILF_L__fold_0__Apss__TractoDim",
    # "Task741__T1__ILF_L__fold_1__Apss__TractoDim",
    # "Task742__T1__ILF_L__fold_2__Apss__TractoDim",
    # "Task743__T1__ILF_L__fold_3__Apss__TractoDim",
    # "Task744__T1__ILF_L__fold_4__Apss__TractoDim",
    #     "Task745__AP__ILF_L__fold_0__Apss__TractoDim",
    # "Task746__AP__ILF_L__fold_1__Apss__TractoDim",
    # "Task747__AP__ILF_L__fold_2__Apss__TractoDim",
    # "Task748__AP__ILF_L__fold_3__Apss__TractoDim",
    # "Task749__AP__ILF_L__fold_4__Apss__TractoDim",
    #     "Task750__peaks__ILF_L__fold_0__Apss__TractoDim",
    # "Task751__peaks__ILF_L__fold_1__Apss__TractoDim",
    # "Task752__peaks__ILF_L__fold_2__Apss__TractoDim",
    # "Task753__peaks__ILF_L__fold_3__Apss__TractoDim",
    # "Task754__peaks__ILF_L__fold_4__Apss__TractoDim",
    # "Task725__T1__IFOF_L__fold_0__Apss__TractoDim",
    # "Task726__T1__IFOF_L__fold_1__Apss__TractoDim",
    # "Task727__T1__IFOF_L__fold_2__Apss__TractoDim",
    # "Task728__T1__IFOF_L__fold_3__Apss__TractoDim",
    # "Task729__T1__IFOF_L__fold_4__Apss__TractoDim",
    #     "Task730__AP__IFOF_L__fold_0__Apss__TractoDim",
    # "Task731__AP__IFOF_L__fold_1__Apss__TractoDim",
    # "Task732__AP__IFOF_L__fold_2__Apss__TractoDim",
    # "Task733__AP__IFOF_L__fold_3__Apss__TractoDim",
    # "Task734__AP__IFOF_L__fold_4__Apss__TractoDim",
    #     "Task735__peaks__IFOF_L__fold_0__Apss__TractoDim",
    # "Task736__peaks__IFOF_L__fold_1__Apss__TractoDim",
    # "Task737__peaks__IFOF_L__fold_2__Apss__TractoDim",
    # "Task738__peaks__IFOF_L__fold_3__Apss__TractoDim",
    # "Task739__peaks__IFOF_L__fold_4__Apss__TractoDim",
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
    print(len(splits))
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


