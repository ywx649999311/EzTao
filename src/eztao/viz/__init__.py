import matplotlib as mpl
import eztao
import os
from .mpl_viz import plot_drw_ll, plot_dho_ll
from .mpl_viz import plot_pred_lc

mpl.rc_file(os.path.join(eztao.__path__[0], "viz/eztao.rc"))
