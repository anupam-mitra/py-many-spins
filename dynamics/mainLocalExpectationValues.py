import numpy as np
import itertools

import uuid
import time
import itertools
import logging

import tenpy
import tenpy.models
import tenpy.algorithms
import tenpy.simulations
import tenpy.networks
import tenpy.networks.site

import sqlite3

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

import sys
sys.path.append("../src")
logging.info(sys.path)

from wrap_tenpy.TenPy_OnlineCompress import *
from wrap_tenpy.distancemeasures import hilbertschmidt_distance

####################################################################################################
if __name__ == '__main__':
    string_date:str = "2023-058_13-37-35"
    n_spins:int = 20

    string_date:str = "a33cdc3b-dd7b-4ad5-aaa2-79bd3d4f0969"
    string_date:str = sys.argv[1]
    n_spins:int = 40

    logging.info("string_date = %s" % (string_date,))
    logging.info("n_spins = %d" % (n_spins,))
    
    filename_mps_df = "pkl/mps/%s_tebd_mps.pkl" % (string_date)

    READ_FLAGS = "rb"
    
    with open(filename_mps_df, READ_FLAGS) as iofile:
        df_mps_all = pickle.load(iofile)
        
    logging.info("df_mps_all = \n%s" % (df_mps_all,))

    bonddim_list = df_mps_all["bonddim"].sort_values().unique()
    ix_time_list = df_mps_all["ix_time"].sort_values().unique()
    time_list = df_mps_all["time"].sort_values().unique()

    logging.info("bonddim_list = %s" % bonddim_list)
    logging.info("ix_time_list = %s" % ix_time_list)

    if False:
        df_1spin_expect = df_mps_all[df_mps_all.columns.drop("mps").drop("walltime")]
        mps = df_mps_all["mps"]

        walltime_begin = time.time()
    
        expect_zz = mps.apply(lambda s: s.correlation_function("Sigmaz", "Sigmaz"))
        #expect_x = mps.apply(lambda s: s.expectation_value("Sigmax"))
        #expect_y = mps.apply(lambda s: s.expectation_value("Sigmay"))
        #expect_z = mps.apply(lambda s: s.expectation_value("Sigmaz"))

        walltime_end = time.time()
        walltime_duration = walltime_end - walltime_begin
        logging.info("Time taken = %g s" % (walltime_duration))

        #df_1spin_expect["expect_x"] = expect_x
        #df_1spin_expect["expect_y"] = expect_y
        df_1spin_expect["expect_zz"] = expect_zz
        
        logging.info("df_1spin_expect = \n%s" % (df_1spin_expect))
        
        WRITE_FLAGS = "wb"

        filename_mps_df = "pkl/expectation/%s_tebd_2spin.pkl" % (string_date)
        with open(filename_mps_df, WRITE_FLAGS) as iofile:
            pickle.dump(df_1spin_expect, iofile)
            
            logging.info("Comparing")

            for bonddim_low, bonddim_high in zip(bonddim_list[:-1], bonddim_list[1:]):

                df_low = df_1spin_expect[df_1spin_expect["bonddim"] == bonddim_low]
                df_high = df_1spin_expect[df_1spin_expect["bonddim"] == bonddim_high]

                expect_low = df_low["expect_z"].to_numpy()
                expect_high = df_high["expect_z"].to_numpy()
        
                expect_diff_sq = (expect_high - expect_low)**2
        
                logging.info("bonddim_low = %d" % bonddim_low)
                logging.info("bonddim_high = %d" % bonddim_high)
                logging.info("expect_diff_sq = %s" % (expect_diff_sq))
            
    logging.info("Evaluating 1-spin reduced density operators")
    
    walltime_begin = time.time()
    rows_1spin_reduced_dm = []

    sites_sel_list = [s for s in itertools.combinations(range(n_spins), 2)]
    logging.info("sites_set_list = %s" % sites_sel_list)
    
    for row_iteration in df_mps_all.itertuples():
        for sites_sel in sites_sel_list:

            logging.info("Reduced state for %s in %d at ix_time = %d" \
                         % (sites_sel, n_spins, row_iteration.ix_time))
            mps = row_iteration.mps

            rho = mps.get_rho_segment(sites_sel)

            row_created = {
                "ix_time": row_iteration.ix_time,
                "time": row_iteration.ix_time,
                "bonddim": row_iteration.bonddim,
                "sites_sel": sites_sel,
                "rho": rho,
            }

            rows_1spin_reduced_dm.append(row_created)
            
    df_1spin_reduced_dm = pandas.DataFrame(rows_1spin_reduced_dm)

    walltime_end = time.time()
    walltime_duration = walltime_end - walltime_begin
    logging.info("Time taken = %g s" % (walltime_duration))

    logging.info(df_1spin_reduced_dm)

    logging.info("Comparing")

    rows_hsd = []
    for bonddim_low, bonddim_high in zip(bonddim_list[:-1], bonddim_list[1:]):
        for sites_sel in sites_sel_list:

            df_low = df_1spin_reduced_dm[ \
                                          (df_1spin_reduced_dm["bonddim"] == bonddim_low) & \
                                          (df_1spin_reduced_dm["sites_sel"] == sites_sel) \
                                         ]
            df_high = df_1spin_reduced_dm[ \
                                           (df_1spin_reduced_dm["bonddim"] == bonddim_high) & \
                                           (df_1spin_reduced_dm["sites_sel"] == sites_sel) \
                                          ]
        
            rho_low_list = df_low["rho"].to_numpy()
            rho_high_list = df_high["rho"].to_numpy()
        
            for ix_time in ix_time_list:
            
                rho_low = rho_low_list[ix_time]
                rho_high = rho_high_list[ix_time]

                hsd = hilbertschmidt_distance(rho_low, rho_high)
                
                # Saving sites_sel as a single value for compability with SQL
                # Only works for 1-spin reduced density operators
                row_created = {
                    "ix_time": ix_time,
                    "time": time_list[ix_time],
                    "bonddim_low": bonddim_low,
                    "bonddim_high": bonddim_high,
                    "sites_sel": "%s" % (sites_sel,),
                    "hsd": hsd,
                }

                rows_hsd.append(row_created)

            logging.info("bonddim_low = %d" % bonddim_low)
            logging.info("bonddim_high = %d" % bonddim_high)
            logging.info("sites_sel = %s" % (sites_sel,))

    df_hsd = pandas.DataFrame(rows_hsd)

    logging.info("df_hsd = \n%s" % df_hsd)
    logging.info("Maximum hsd = %d" % np.max(df_hsd["hsd"]))
    logging.info("at \n%s" % \
                 df_hsd[df_hsd["hsd"] == np.max(df_hsd["hsd"])])

    logging.info("Saving to SQL database")
    connection = sqlite3.connect("hsd.db")
    df_hsd.to_sql("HSD", connection, if_exists="replace")

    connection.close()
    logging.info("Saved to SQL database")
