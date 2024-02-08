from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from nnunet.paths import preprocessing_output_dir

tasks = [
    # "Task822__T1__PYT_L__TractoInferno",
    # "Task823__AP__PYT_L__TractoInferno",
    # "Task824__peaks__PYT_L__TractoInferno",
    # "Task825__T1__PT_L__fold_0__Apss__TractoDim",
    # "Task826__T1__PT_L__fold_1__Apss__TractoDim",
    # "Task827__T1__PT_L__fold_2__Apss__TractoDim",
    # "Task828__T1__PT_L__fold_3__Apss__TractoDim",
    # "Task829__T1__PT_L__fold_4__Apss__TractoDim",
    # "Task830__AP__PT_L__fold_0__Apss__TractoDim",
    # "Task831__AP__PT_L__fold_1__Apss__TractoDim",
    # "Task832__AP__PT_L__fold_2__Apss__TractoDim",
    # "Task833__AP__PT_L__fold_3__Apss__TractoDim",
    # "Task834__AP__PT_L__fold_4__Apss__TractoDim",
    # "Task835__peaks__PT_L__fold_0__Apss__TractoDim",
    # "Task836__peaks__PT_L__fold_1__Apss__TractoDim",
    # "Task837__peaks__PT_L__fold_2__Apss__TractoDim",
    # "Task838__peaks__PT_L__fold_3__Apss__TractoDim",
    # "Task839__peaks__PT_L__fold_4__Apss__TractoDim",
    #  "Task840__T1__AF_L__TractoInferno",
    # "Task841__AP__AF_L__TractoInferno",
    # "Task842__peaks__AF_L__TractoInferno",
    # "Task843__T1__AF_L__fold_0__Apss__TractoDim",
    # "Task844__T1__AF_L__fold_1__Apss__TractoDim",
    # "Task845__T1__AF_L__fold_2__Apss__TractoDim",
    # "Task846__T1__AF_L__fold_3__Apss__TractoDim",
    # "Task847__T1__AF_L__fold_4__Apss__TractoDim",
    # "Task848__AP__AF_L__fold_0__Apss__TractoDim",
    # "Task849__AP__AF_L__fold_1__Apss__TractoDim",
    # "Task850__AP__AF_L__fold_2__Apss__TractoDim",
    # "Task851__AP__AF_L__fold_3__Apss__TractoDim",
    # "Task852__AP__AF_L__fold_4__Apss__TractoDim",
    # "Task853__peaks__AF_L__fold_0__Apss__TractoDim",
    #"Task854__peaks__AF_L__fold_1__Apss__TractoDim",
    # "Task855__peaks__AF_L__fold_2__Apss__TractoDim",
    #"Task856__peaks__AF_L__fold_3__Apss__TractoDim",
    # "Task857__peaks__AF_L__fold_4__Apss__TractoDim",
    # "Task858__T1__FAT_L__TractoInferno",
    # "Task859__AP__FAT_L__TractoInferno",
    # "Task860__peaks__FAT_L__TractoInferno",
    # "Task861__T1__FAT_L__fold_0__Apss__TractoDim",
    # "Task862__T1__FAT_L__fold_1__Apss__TractoDim",
    # "Task863__T1__FAT_L__fold_2__Apss__TractoDim",
    # "Task864__T1__FAT_L__fold_3__Apss__TractoDim",
    # "Task865__T1__FAT_L__fold_4__Apss__TractoDim",
    # "Task866__AP__FAT_L__fold_0__Apss__TractoDim",
    # "Task867__AP__FAT_L__fold_1__Apss__TractoDim",
    # "Task868__AP__FAT_L__fold_2__Apss__TractoDim",
    # "Task869__AP__FAT_L__fold_3__Apss__TractoDim",
    # "Task870__AP__FAT_L__fold_4__Apss__TractoDim",
    # "Task871__peaks__FAT_L__fold_0__Apss__TractoDim",
    # "Task872__peaks__FAT_L__fold_1__Apss__TractoDim",
    # "Task873__peaks__FAT_L__fold_2__Apss__TractoDim",
    # "Task874__peaks__FAT_L__fold_3__Apss__TractoDim",
    # "Task875__peaks__FAT_L__fold_4__Apss__TractoDim",
# "Task876__T1__ILF_L__TractoInferno",
# "Task877__AP__ILF_L__TractoInferno",
# "Task878__peaks__ILF_L__TractoInferno",
# "Task879__T1__ILF_L__fold_0__Apss__TractoDim",
# "Task880__T1__ILF_L__fold_1__Apss__TractoDim",
# "Task881__T1__ILF_L__fold_2__Apss__TractoDim",
# "Task882__T1__ILF_L__fold_3__Apss__TractoDim",
# "Task883__T1__ILF_L__fold_4__Apss__TractoDim",
# "Task884__AP__ILF_L__fold_0__Apss__TractoDim",
# "Task885__AP__ILF_L__fold_1__Apss__TractoDim",
# "Task886__AP__ILF_L__fold_2__Apss__TractoDim",
# "Task887__AP__ILF_L__fold_3__Apss__TractoDim",
# "Task888__AP__ILF_L__fold_4__Apss__TractoDim",
# "Task889__peaks__ILF_L__fold_0__Apss__TractoDim",
# "Task890__peaks__ILF_L__fold_1__Apss__TractoDim",
# "Task891__peaks__ILF_L__fold_2__Apss__TractoDim",
# "Task892__peaks__ILF_L__fold_3__Apss__TractoDim",
# "Task893__peaks__ILF_L__fold_4__Apss__TractoDim",
# "Task894__T1__IFOF_L__TractoInferno",
# "Task895__AP__IFOF_L__TractoInferno",
# "Task896__peaks__IFOF_L__TractoInferno",
# "Task897__T1__IFOF_L__fold_0__Apss__TractoDim",
# "Task898__T1__IFOF_L__fold_1__Apss__TractoDim",
# "Task899__T1__IFOF_L__fold_2__Apss__TractoDim",
# "Task900__T1__IFOF_L__fold_3__Apss__TractoDim",
# "Task901__T1__IFOF_L__fold_4__Apss__TractoDim",
# "Task902__AP__IFOF_L__fold_0__Apss__TractoDim",
# "Task903__AP__IFOF_L__fold_1__Apss__TractoDim",
# "Task904__AP__IFOF_L__fold_2__Apss__TractoDim",
# "Task905__AP__IFOF_L__fold_3__Apss__TractoDim",
# "Task906__AP__IFOF_L__fold_4__Apss__TractoDim",
# "Task907__peaks__IFOF_L__fold_0__Apss__TractoDim",
# "Task908__peaks__IFOF_L__fold_1__Apss__TractoDim",
# "Task909__peaks__IFOF_L__fold_2__Apss__TractoDim",
# "Task910__peaks__IFOF_L__fold_3__Apss__TractoDim",
# "Task911__peaks__IFOF_L__fold_4__Apss__TractoDim",
# "Task912__T1__UF_L__TractoInferno",
# "Task913__AP__UF_L__TractoInferno",
# "Task914__peaks__UF_L__TractoInferno",
# "Task915__T1__UF_L__fold_0__Apss__TractoDim",
# "Task916__T1__UF_L__fold_1__Apss__TractoDim",
# "Task917__T1__UF_L__fold_2__Apss__TractoDim",
# "Task918__T1__UF_L__fold_3__Apss__TractoDim",
# "Task919__T1__UF_L__fold_4__Apss__TractoDim",
# "Task920__AP__UF_L__fold_0__Apss__TractoDim",
# "Task921__AP__UF_L__fold_1__Apss__TractoDim",
# "Task922__AP__UF_L__fold_2__Apss__TractoDim",
# "Task923__AP__UF_L__fold_3__Apss__TractoDim",
# "Task924__AP__UF_L__fold_4__Apss__TractoDim",
# "Task925__peaks__UF_L__fold_0__Apss__TractoDim",
# "Task926__peaks__UF_L__fold_1__Apss__TractoDim",
# "Task927__peaks__UF_L__fold_2__Apss__TractoDim",
# "Task928__peaks__UF_L__fold_3__Apss__TractoDim",
# "Task929__peaks__UF_L__fold_4__Apss__TractoDim"

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


