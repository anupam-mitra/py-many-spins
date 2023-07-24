import numpy as np

import uuid
import time
import itertools
import logging
import os
import pickle
import pandas
import sqlite3

import tenpy
import tenpy.models
import tenpy.algorithms
import tenpy.simulations
import tenpy.networks
import tenpy.networks.site

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
        level=logging.INFO)

import sys
sys.path.append("../src")
logging.info(sys.path)

import config
from wrap_tenpy.distancemeasures import hilbertschmidt_distance

####################################################################################################
if __name__ == '__main__':

    try:
        string_uuid_a:str = sys.argv[1]
        string_uuid_b:str = sys.argv[2]
    
        bonddim_a:int = int(sys.argv[3])
        bonddim_b:int = int(sys.argv[4])
    
        n_spins:int = int(sys.argv[5])
        size_marginal:int = int(sys.argv[6])

    except Exception as e:
        print("Usage: %s <uuid_a> <uuid_b> <bonddim_a> <bonddim_b>" + \
            "<n_spins> <marginal_size>"\
                % (sys.argv[0],))

        exit(-1)

    logging.info("config.data_directory = %s" % (config.data_directory))
    logging.info("config.index_directory = %s" % (config.index_directory))
    logging.info("config.reducedstate_directory = %s" % (config.reducedstate_directory))
    logging.info("config.mps_directory = %s" % (config.mps_directory))
    
    logging.info("string_uuid_a = %s" % (string_uuid_a,))
    logging.info("string_uuid_b = %s" % (string_uuid_b,))
    logging.info("bonddim_a = %s" % (bonddim_a,))
    logging.info("bonddim_b = %s" % (bonddim_b,))
    logging.info("n_spins = %d" % (n_spins,))

    filename_df_a = os.path.join(
            config.reducedstate_directory, "%s_%d-spin.pkl" % \
                    (string_uuid_a, size_marginal))

    filename_df_b = os.path.join(
            config.reducedstate_directory, "%s_%d-spin.pkl" % \
                    (string_uuid_b, size_marginal))

    READ_FLAGS = "rb"

    with open(filename_df_a, READ_FLAGS) as iofile:
        df_a = pickle.load(iofile)
        
    with open(filename_df_b, READ_FLAGS) as iofile:
        df_b = pickle.load(iofile)
        
    logging.info("df_a = \n%s" % (df_a,))
    logging.info("df_b = \n%s" % (df_b,))

    logging.info("Comparing bonddim = %d with bonddim = %d" % \
            (bonddim_a, bonddim_b))

    walltime_begin:float = time.time()

    entries_hsd = []
    for entry_a, entry_b in zip(df_a.itertuples(), df_b.itertuples()):
       
        rho_a = entry_a.rho
        rho_b = entry_b.rho

        sqhsd = hilbertschmidt_distance(rho_a, rho_b)

        entry_current = {
            'ix_time_a': entry_a.ix_time,
            'time_a': entry_a.time,
            'ix_time_b': entry_b.ix_time,
            'time_b': entry_b.time,
            'sites_sel_a': entry_a.sites_sel,
            'sites_sel_b': entry_b.sites_sel,
            'bonddim_a': bonddim_a,
            'bonddim_b': bonddim_b,
            'sqhsd': sqhsd,
         }

        entries_hsd.append(entry_current)

    df_hsd = pandas.DataFrame(entries_hsd)

    logging.info("df_hsd = \n%s" % (df_hsd.sort_values(by='sites_sel_a'),))

    walltime_end:float = time.time()
    walltime_duration:float = walltime_end - walltime_begin
    logging.info("Time taken = %g s" % (walltime_duration))
