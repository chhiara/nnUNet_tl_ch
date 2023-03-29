from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from nnunet.paths import preprocessing_output_dir
task_name = 'Task601__T1__AF_L__TractoInferno'

# make sure to run the Task601 dataset conversion and
# nnUNet_plan_and_preprocess first!
plans_fname = join(preprocessing_output_dir, task_name, 'nnUNetPlansv2.1_plans_3D.pkl')
plans = load_pickle(plans_fname)
plans['plans_per_stage'][0]['patch_size'] = np.array((32, 32, 32))
plans['plans_per_stage'][0]['batch_size'] = 48
#plans['plans_per_stage'][0]['num_pool_per_axis'] = [7, 7]
# because we changed the num_pool_per_axis, we need to change conv_kernel_sizes and pool_op_kernel_sizes as well!
#plans['plans_per_stage'][0]['pool_op_kernel_sizes'] = [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
#plans['plans_per_stage'][0]['conv_kernel_sizes'] = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
# for a network with num_pool_per_axis [7,7] the correct length of pool kernel sizes is 7 and the length of conv
# kernel sizes is 8! Note that you can also change these numbers if you believe it makes sense. A pool kernel size
# of 1 will result in no pooling along that axis, a kernel size of 3 will reduce the size of the feature map
# representations by factor 3 instead of 2.

# save the plans under a new plans name. Note that the new plans file must end with _plans_2D.pkl!
save_pickle(plans, join(preprocessing_output_dir, task_name, 'nnUNetPlansv2.1_ps32_bs48_plans_3D.pkl'))
