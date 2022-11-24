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

draw_accuracy_cmp("fed3-opt-ACC-comp-2022-11-24-16-23", "fed3-1103-ACC-2022-11-24-16-21",
                  "fedopt-1103-ACC-2022-11-24-16-03", "../../../OutputData/fed_3", "../../../OutputData/fed_opt")

draw_loss_cmp("fed3-opt-LOSS-comp-2022-11-24-16-37", "fed3-1103-LOSS-2022-11-24-16-21",
              "fedopt-1103-LOSS-2022-11-24-16-03", "../../../OutputData/fed_3", "../../../OutputData/fed_opt")

draw_time_cmp("fed3-opt-TIME-comp-2022-11-24-18-13", "fed3-1103-TIME-2022-11-24-17-43",
              "fedopt-1103-TIME-2022-11-24-18-04", "../../../OutputData/fed_3", "../../../OutputData/fed_opt")

# draw_goal_cmp("fed3-opt-GOAL-comp-2022-11-24-18-13", "fed3-1103-TIME-2022-11-24-17-43",
# "fedopt-1103-TIME-2022-11-24-18-04", "../../../OutputData/fed_3", "../../../OutputData/fed_opt")
draw_training_intensity_sum_cmp("fed3-opt-TI-comp-2022-11-24-18-13", "fed3-1103-TIME-2022-11-24-17-43",
                                "fedopt-1103-TIME-2022-11-24-18-04", "../../../OutputData/fed_3",
                                "../../../OutputData/fed_opt")

draw_goal_cmp("fed3-opt-GOAL-comp-2022-11-24-18-13", filename_3, filename_opt, DATA_PATH_3, DATA_PATH_OPT)
