import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.utils.draw import draw_loss
from fedml_api.utils.draw import draw_accuracy

draw_accuracy("fed3-1103-ACC-2022-11-23-22-28")
draw_loss("fed3-1103-LOSS-2022-11-23-22-28")

