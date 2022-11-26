import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.utils.draw import *

# draw_accuracy("fed3-1103-ACC-2022-11-23-22-28")
# draw_loss("fed3-1103-LOSS-2022-11-23-22-28")

draw_accuracy_cmp()
draw_loss_cmp()
draw_training_intensity_sum_cmp()
draw_time_cmp()
draw_goal_cmp()

# draw_accuracy_budget()
# draw_loss_budget()
# draw_time_budget()

# draw_accuracy_cmp_with_time()
# draw_loss_cmp_with_time()
