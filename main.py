import os
import time
from DriveTestAnalysis.scripts.func import calcBearing, CalcDistanceKM, stateOpt, CellState
from DriveTestAnalysis.scripts.calculations import calc, CalcDistanceKM, Missing
from DriveTestAnalysis.scripts.FinalPilot import fn_CalcParcelID, pilot
from DriveTestAnalysis.scripts.LTE_data_analysis import LTE
# from DriveTestAnalysis.scripts.sector_percentage import create_sector_perecentage
# from DriveTestAnalysis.scripts.States_map import generate_heat_map
from multiprocessing import Process
from django.core.files import File
import pandas as pd
import numpy as np
import csv
from pandas import DataFrame
from math import *
import simplekml
from math import ceil


def main_function1(Data1, TRAIN):
    print("Main Start")
    print(Data1)
    print(TRAIN)
    LTE(Data1, TRAIN) #stuck
    print("After LTE")
    test_output_file = open('TestDataset.csv')
    print(test_output_file)
    return File(test_output_file)









def main_function(drive, scanner, cell):
    # print(pd.read_csv(drive))
    # print(pd.read_csv(scanner))
    # print(pd.read_csv(cell))
    start_time = time.time()
    CellState(cell, drive)
    Missing(cell, drive)
    pilot(cell, scanner)

    #  p1 = Process(target=CellState, args=(cell, drive))
    # p1.start()
    # p2 = Process(target=Missing, args=(cell, drive))
    # p2.start()
    # p3 = Process(target=pilot, args=(cell, scanner))
    # p3.start()
    # p1.join()
    # p2.join()
    # p3.join()
    end_time = time.time()
    time_diff = end_time - start_time

    # os.system("python scripts > test_output_file.csv")
    test_output_file1 = open('Missing_NBR_Analysis_test.csv')
    test_output_file2 = open('final_data_pilot.csv')
    test_output_file3 = open('Final_Report.csv')

    return File(test_output_file1), File(test_output_file2), File(test_output_file3)

# def heatt_map(Final_Report, serving_cell_is_back_lobe, Diff_Sector):
#   create_sector_perecentage(Final_Report, serving_cell_is_back_lobe, Diff_Sector)


# def heatt_map(Final_Report, total_box, info_box):
#   generate_heat_map(Final_Report, total_box, info_box)
