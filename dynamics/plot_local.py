import numpy as np
import itertools

import uuid
import time
import itertools
import logging
import pandas

import tenpy
import tenpy.models
import tenpy.algorithms
import tenpy.simulations
import tenpy.networks
import tenpy.networks.site

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

import sys
sys.path.append("../src")
logging.info(sys.path)

from wrap_tenpy.TenPy_OnlineCompress import *
from wrap_tenpy.distancemeasures import hilbertschmidt_distance

import matplotlib.pyplot as plt

if __name__ == "__main__":

    string_date = "2023-040_18-53-29"
    n_spins:int = 100

    string_date = "2023-037_10-40-23"
    n_spins:int = 40


    READ_FLAGS = "rb"

    filename_mps_df = "pkl/expectation/%s_tebd_2spin.pkl" % (string_date)
    with open(filename_mps_df, READ_FLAGS) as iofile:
        df_1spin_expect = pickle.load(iofile)


    logging.info("df_1spin_expect = %s\n"  % df_1spin_expect)

        
    bonddim_list = df_1spin_expect["bonddim"].sort_values().unique()
    ix_time_list = df_1spin_expect["ix_time"].sort_values().unique()
    time_list = df_1spin_expect["time"].sort_values().unique()

    ix_time_selected = [0, 15, 31, 63, 127, 255]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    
    for ix_ix_time, ix_time in enumerate(ix_time_selected):

        logging.info("ix_time = %d" % ix_time)
        logging.info("df_1spin_expect.columns = %s" % (df_1spin_expect.columns))
        df_sel = df_1spin_expect.loc[(df_1spin_expect["ix_time"] == ix_time)]
        #df_sel = df_1spin_expect

        ax.plot(df_sel["bonddim"], 
		[np.mean(xi) for xi in df_sel["expect_zz"].to_numpy()], 
                marker='o', label=r"$Jt=%0.2f$" % time_list[ix_time])

    ax.set_yscale("log")
    ax.legend(ncol=2)
    ax.set_xlabel("Bond-dimension")
    ax.set_ylabel(r"$ 2/n(n-1) \sum_{\ell, \ell'} \langle \sigma^z_{\ell} \sigma^{z}_{\ell^\prime} \rangle$")
    ax.set_title(r"$|\uparrow_z\rangle^{\otimes %d} \to J = B_{\perp}$" % (n_spins,))
    plt.savefig("plots/___.pdf")
    #plt.show()


        
