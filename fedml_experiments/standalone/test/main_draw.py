import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.utils.draw import *

filename_3 = "fed3-1103-INFO-2022-11-24-19-25"
filename_opt = ""
DATA_PATH_3 = "../../../OutputData/fed_3"
DATA_PATH_OPT = "../../../OutputData/fed_opt"

# draw_accuracy("fed3-1103-ACC-2022-11-23-22-28")
# draw_loss("fed3-1103-LOSS-2022-11-23-22-28")

draw_accuracy_cmp()
draw_loss_cmp()
draw_training_intensity_sum_cmp()
draw_time_cmp()
draw_goal_cmp()
