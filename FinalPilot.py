# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:41:53 2020

@author: Heba Ramadan
"""

# lw attached b aktr mn 3 de pilot problem

import os  # to use fns of OS
import pandas as pd  # for data manipulation and analysis
import numpy  # n dimensionnal array object
import csv  # Comma separated values
import groupbytime
from math import *
import numpy as np
from pandas import DataFrame
import time


def CalcDistanceKM(a_lat, a_lng, b_lat, b_lng):
    R = 6371  # earth radius in km
    a_lat = np.radians(a_lat)
    a_lng = np.radians(a_lng)
    b_lat = np.radians(b_lat)
    b_lng = np.radians(b_lng)
    d_lat = b_lat - a_lat
    d_lng = b_lng - a_lng
    d_lat_sq = np.sin(d_lat / 2) ** 2
    d_lng_sq = np.sin(d_lng / 2) ** 2
    a = d_lat_sq + np.cos(a_lat) * np.cos(b_lat) * d_lng_sq
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def fn_CalcParcelID(Pos, ParcelUnitSize):
    if (Pos == 500):  # null parcel
        Result = int(50000000)
    elif (Pos < 0):
        Result = int(Pos * 100000) - ParcelUnitSize + (int(Pos * 100000) % ParcelUnitSize)
    else:
        Result = int(Pos * 100000) - (int(Pos * 100000) % ParcelUnitSize)
    return int(Result)


def pilot(x, y):
    ParcelSize = 50
    # dividing area into squares lkol lat w lon
    UARFCN = 3087
    # UTRA Absolute Radio Frequency Channel Number to calculate carrier freq
    # read data from csv file
    # Y= pilot file (scan bdl el drive test file) X=cell file
    scanner_File = pd.read_csv(y)
    Cell_file = pd.read_csv(x)
    Cells = Cell_file[['Cell', 'Lat', 'Lon', 'SC', 'UARFCN', 'ANT_DIRECTION']]
    # PSC primary scrambling code
    # sc aggr ec CPICH_Scan_RSCP_SortedBy_EcIo
    # EC/IO is a measure of the quality/cleanliness of the signal from the tower to the modem and indicates the signal-tonoise ratio (the ratio of the received/good energy to the interference/bad energy). It is measured in decibels (dB).
    scanner = scanner_File[
        ['Latitude', 'Longitude', 'PSC: Top #1 (UARFCN #01)', 'Sc Aggr Ec (dBm): Top #1 (UARFCN #01)',
         'Sc Aggr Ec/Io (dB): Top #1 (UARFCN #01)',
         'PSC: Top #2 (UARFCN #01)', 'Sc Aggr Ec (dBm): Top #2 (UARFCN #01)', 'Sc Aggr Ec/Io (dB): Top #2 (UARFCN #01)',
         'PSC: Top #3 (UARFCN #01)', 'Sc Aggr Ec (dBm): Top #3 (UARFCN #01)', 'Sc Aggr Ec/Io (dB): Top #3 (UARFCN #01)',
         'PSC: Top #4 (UARFCN #01)', 'Sc Aggr Ec (dBm): Top #4 (UARFCN #01)', 'Sc Aggr Ec/Io (dB): Top #4 (UARFCN #01)',
         'PSC: Top #5 (UARFCN #01)', 'Sc Aggr Ec (dBm): Top #5 (UARFCN #01)', 'Sc Aggr Ec/Io (dB): Top #5 (UARFCN #01)',
         'PSC: Top #6 (UARFCN #01)', 'Sc Aggr Ec (dBm): Top #6 (UARFCN #01)', 'Sc Aggr Ec/Io (dB): Top #6 (UARFCN #01)',
         'PSC: Top #7 (UARFCN #01)', 'Sc Aggr Ec (dBm): Top #7 (UARFCN #01)', 'Sc Aggr Ec/Io (dB): Top #7 (UARFCN #01)',
         'PSC: Top #8 (UARFCN #01)', 'Sc Aggr Ec (dBm): Top #8 (UARFCN #01)', 'Sc Aggr Ec/Io (dB): Top #8 (UARFCN #01)',
         'PSC: Top #9 (UARFCN #01)', 'Sc Aggr Ec (dBm): Top #9 (UARFCN #01)',
         'Sc Aggr Ec/Io (dB): Top #9 (UARFCN #01)']]
    scanner_size = scanner.shape[0]  # 1st dimension of array
    cells_size = Cells.shape[0]
    ## data frame:two-dimensional tabular data structure with labeled axes (rows and columns).
    A1 = pd.DataFrame(columns=['Latitude', 'Longitude', 'PSC', 'EcNo', 'RSCP'])
    A2 = pd.DataFrame(columns=['Latitude', 'Longitude', 'PSC', 'EcNo', 'RSCP'])
    A3 = pd.DataFrame(columns=['Latitude', 'Longitude', 'PSC', 'EcNo', 'RSCP'])
    A4 = pd.DataFrame(columns=['Latitude', 'Longitude', 'PSC', 'EcNo', 'RSCP'])
    A5 = pd.DataFrame(columns=['Latitude', 'Longitude', 'PSC', 'EcNo', 'RSCP'])
    A6 = pd.DataFrame(columns=['Latitude', 'Longitude', 'PSC', 'EcNo', 'RSCP'])
    A7 = pd.DataFrame(columns=['Latitude', 'Longitude', 'PSC', 'EcNo', 'RSCP'])
    A8 = pd.DataFrame(columns=['Latitude', 'Longitude', 'PSC', 'EcNo', 'RSCP'])
    A9 = pd.DataFrame(columns=['Latitude', 'Longitude', 'PSC', 'EcNo', 'RSCP'])
    for i in range(scanner_size):
        # if isnan(scanner['PSC: Top #1 (UARFCN #01)'][i]) == False:
        if (scanner['PSC: Top #1 (UARFCN #01)'][i]) != -1:
            A1 = A1.append({'Latitude': scanner['Latitude'][i], 'Longitude': scanner['Longitude'][i],
                            'PSC': scanner['PSC: Top #1 (UARFCN #01)'][i],
                            'EcNo': scanner['Sc Aggr Ec/Io (dB): Top #1 (UARFCN #01)'][i],
                            'RSCP': scanner['Sc Aggr Ec (dBm): Top #1 (UARFCN #01)'][i]}, ignore_index=True)
        if (scanner['PSC: Top #2 (UARFCN #01)'][i]) != -1:
            A2 = A2.append({'Latitude': scanner['Latitude'][i], 'Longitude': scanner['Longitude'][i],
                            'PSC': scanner['PSC: Top #2 (UARFCN #01)'][i],
                            'EcNo': scanner['Sc Aggr Ec/Io (dB): Top #2 (UARFCN #01)'][i],
                            'RSCP': scanner['Sc Aggr Ec (dBm): Top #2 (UARFCN #01)'][i]}, ignore_index=True)
        if (scanner['PSC: Top #3 (UARFCN #01)'][i]) != -1:
            A3 = A3.append({'Latitude': scanner['Latitude'][i], 'Longitude': scanner['Longitude'][i],
                            'PSC': scanner['PSC: Top #3 (UARFCN #01)'][i],
                            'EcNo': scanner['Sc Aggr Ec/Io (dB): Top #3 (UARFCN #01)'][i],
                            'RSCP': scanner['Sc Aggr Ec (dBm): Top #3 (UARFCN #01)'][i]}, ignore_index=True)
        if (scanner['PSC: Top #4 (UARFCN #01)'][i]) != -1:
            A4 = A4.append({'Latitude': scanner['Latitude'][i], 'Longitude': scanner['Longitude'][i],
                            'PSC': scanner['PSC: Top #4 (UARFCN #01)'][i],
                            'EcNo': scanner['Sc Aggr Ec/Io (dB): Top #4 (UARFCN #01)'][i],
                            'RSCP': scanner['Sc Aggr Ec (dBm): Top #4 (UARFCN #01)'][i]}, ignore_index=True)
        if (scanner['PSC: Top #5 (UARFCN #01)'][i]) != -1:
            A5 = A5.append({'Latitude': scanner['Latitude'][i], 'Longitude': scanner['Longitude'][i],
                            'PSC': scanner['PSC: Top #5 (UARFCN #01)'][i],
                            'EcNo': scanner['Sc Aggr Ec/Io (dB): Top #5 (UARFCN #01)'][i],
                            'RSCP': scanner['Sc Aggr Ec (dBm): Top #5 (UARFCN #01)'][i]}, ignore_index=True)
        if (scanner['PSC: Top #6 (UARFCN #01)'][i]) != -1:
            A6 = A6.append({'Latitude': scanner['Latitude'][i], 'Longitude': scanner['Longitude'][i],
                            'PSC': scanner['PSC: Top #6 (UARFCN #01)'][i],
                            'EcNo': scanner['Sc Aggr Ec/Io (dB): Top #6 (UARFCN #01)'][i],
                            'RSCP': scanner['Sc Aggr Ec (dBm): Top #6 (UARFCN #01)'][i]}, ignore_index=True)
        if (scanner['PSC: Top #7 (UARFCN #01)'][i]) != -1:
            A7 = A7.append({'Latitude': scanner['Latitude'][i], 'Longitude': scanner['Longitude'][i],
                            'PSC': scanner['PSC: Top #7 (UARFCN #01)'][i],
                            'EcNo': scanner['Sc Aggr Ec/Io (dB): Top #7 (UARFCN #01)'][i],
                            'RSCP': scanner['Sc Aggr Ec (dBm): Top #7 (UARFCN #01)'][i]}, ignore_index=True)
        if (scanner['PSC: Top #8 (UARFCN #01)'][i]) != -1:
            A8 = A8.append({'Latitude': scanner['Latitude'][i], 'Longitude': scanner['Longitude'][i],
                            'PSC': scanner['PSC: Top #8 (UARFCN #01)'][i],
                            'EcNo': scanner['Sc Aggr Ec/Io (dB): Top #8 (UARFCN #01)'][i],
                            'RSCP': scanner['Sc Aggr Ec (dBm): Top #8 (UARFCN #01)'][i]}, ignore_index=True)
        if (scanner['PSC: Top #9 (UARFCN #01)'][i]) != -1:
            A9 = A9.append({'Latitude': scanner['Latitude'][i], 'Longitude': scanner['Longitude'][i],
                            'PSC': scanner['PSC: Top #9 (UARFCN #01)'][i],
                            'EcNo': scanner['Sc Aggr Ec/Io (dB): Top #9 (UARFCN #01)'][i],
                            'RSCP': scanner['Sc Aggr Ec (dBm): Top #9 (UARFCN #01)'][i]}, ignore_index=True)
    A = pd.concat([A1, A2, A3, A4, A5, A6, A7, A8, A9], sort=False)
    A = A[~A[['Latitude', 'Longitude', 'PSC', 'EcNo', 'RSCP']].apply(frozenset,
                                                                     axis=1).duplicated()]  # ~ is bitwise not frozenset elem remain unchanged after creation
    # A.to_csv('table_data_pilot.csv',index=True)
    # A = pd.read_csv('table_data_pilot.csv')
    # A=A.iloc[:50,:].reset_index()
    A_size = A.shape[0]
    B1 = pd.DataFrame(columns=['Lat', 'Lon', 'UARFCN', 'PSC', 'SC_Avg_EcNo', 'SC_Avg_RSCP'])
    rows_to_add = pd.DataFrame({
        'Lat': A['Latitude'].apply(
            lambda x: fn_CalcParcelID(x, ParcelSize) / 100000.0
        ),
        'Lon': A['Longitude'].apply(
            lambda x: fn_CalcParcelID(x, ParcelSize) / 100000.0
        ),
        'PSC': A['PSC'],
        'UARFCN': UARFCN,
        'SC_Avg_EcNo': A['EcNo'],
        'SC_Avg_RSCP': A['RSCP'],
    })
    B1 = B1.append(rows_to_add, ignore_index=True)
    B1.to_csv('B1.csv')
    B1 = pd.read_csv('B1.csv')
    B1_size = B1.shape[0]
    #
    Average = B1.groupby(['Lat', 'Lon', 'PSC'])[
        'SC_Avg_EcNo', 'SC_Avg_RSCP'].mean().reset_index()  # grouping data times .mean to calculate avg
    Sum = B1.groupby(['Lat', 'Lon', 'PSC'])
    Sum = Sum.count()  # count of how many times a given object occurs in list
    Sum = Sum.rename(columns={'Unnamed: 0': 'count'})
    Sum = Sum[['count']].reset_index()
    B1 = pd.merge(Sum, Average, how="right")
    B2 = B1.groupby(['Lat', 'Lon'])['SC_Avg_EcNo', 'SC_Avg_RSCP'].max().reset_index()
    B2 = B2.rename(columns={'SC_Avg_EcNo': 'MaxEcNo', 'SC_Avg_RSCP': 'MaxRSCP'})
    B2.to_csv('B2.csv')
    B2_size = B2.shape[0]
    #
    Agg2 = pd.merge(B1, B2, how="inner")
    Agg2.to_csv('Agg2.csv')
    Agg2_size = Agg2.shape[0]

    Agg2_new = pd.DataFrame(columns=['Lat', 'Lon', 'PSC', 'count', 'SC_Avg_EcNo', 'SC_Avg_RSCP', 'MaxEcNo', 'MaxRSCP',
                                     'PilotPollutionFlag'])
    for i in range(Agg2_size):
        if Agg2['SC_Avg_RSCP'][i] > Agg2['MaxRSCP'][i] - 6:
            PilotPollutionFlag = 1
            print (PilotPollutionFlag)
            Agg2_new = Agg2_new.append({'Lat': Agg2['Lat'][i], 'Lon': Agg2['Lon'][i],
                                        'PSC': Agg2['PSC'][i],
                                        'SC_Avg_EcNo': Agg2['SC_Avg_EcNo'][i],
                                        'SC_Avg_RSCP': Agg2['SC_Avg_RSCP'][i],
                                        'MaxRSCP': Agg2['MaxRSCP'][i],
                                        'MaxEcNo': Agg2['MaxEcNo'][i],
                                        'count': Agg2['count'][i],
                                        'PilotPollutionFlag': PilotPollutionFlag

                                        }, ignore_index=True)
        else:
            PilotPollutionFlag = 0
            print (PilotPollutionFlag)
            Agg2_new = Agg2_new.append({'Lat': Agg2['Lat'][i], 'Lon': Agg2['Lon'][i],
                                        'PSC': Agg2['PSC'][i],
                                        'SC_Avg_EcNo': Agg2['SC_Avg_EcNo'][i],
                                        'SC_Avg_RSCP': Agg2['SC_Avg_RSCP'][i],
                                        'MaxRSCP': Agg2['MaxRSCP'][i],
                                        'MaxEcNo': Agg2['MaxEcNo'][i],
                                        'count': Agg2['count'][i],
                                        'PilotPollutionFlag': PilotPollutionFlag

                                        }, ignore_index=True)

    Agg2_new.to_csv('Agg2_new.csv')
    Agg2_new_size = Agg2_new.shape[0]
    #
    final = pd.DataFrame(columns=['Lat', 'Lon', 'PSC', 'count', 'SC_Avg_EcNo', 'SC_Avg_RSCP', 'MaxEcNo', 'MaxRSCP',
                                  'PilotPollutionFlag'])
    for i in range(Agg2_new_size):
        if Agg2_new['SC_Avg_RSCP'][i] >= Agg2_new['MaxRSCP'][i] - 6:
            final = final.append({'Lat': Agg2_new['Lat'][i], 'Lon': Agg2_new['Lon'][i],
                                  'PSC': Agg2_new['PSC'][i],
                                  'SC_Avg_EcNo': Agg2_new['SC_Avg_EcNo'][i],
                                  'SC_Avg_RSCP': Agg2_new['SC_Avg_RSCP'][i],
                                  'MaxRSCP': Agg2_new['MaxRSCP'][i],
                                  'MaxEcNo': Agg2_new['MaxEcNo'][i],
                                  'count': Agg2_new['count'][i],
                                  'PilotPollutionFlag': Agg2_new['PilotPollutionFlag'][i]
                                  }, ignore_index=True)
    # print (final)
    final.to_csv('final.csv')
    final_size = final.shape[0]
    #
    final_table = pd.merge(final, Sum, how="left")
    final_table = final_table.dropna()
    final_table.to_csv('final_data.csv')
    final_table = pd.read_csv('final_data.csv')
    final_table_size = final_table.shape[0]
    Distance = pd.DataFrame(
        columns=['Lat', 'Lon', 'PSC', 'SC_Avg_EcNo', 'SC_Avg_RSCP', 'MaxRSCP', 'MaxEcNo', 'count', 'PilotPollutionFlag',
                 'LatCell', 'LonCell', 'Min_Dist'])
    for i in range(final_table_size):
        B1_sample = DataFrame({'Lat': B1['Lat'][i],
                               'Lon': B1['Lon'][i]}, columns=['Lat', 'Lon'], index=[0])
        B1_sample = B1_sample.iloc[np.full(Cells.shape[0], 0)]
        B1_sample = B1_sample.reset_index()
        Cell1 = Cells
        Cell1['C'] = np.where(
            (abs(B1_sample['Lat'] - Cells['Lat']) < 0.1) & (abs(B1_sample['Lon'] - Cells['Lon']) < 0.1), 'True',
            'False')
        Cell1 = Cell1[Cell1['C'] == "True"].reset_index()
        for j in range(Cell1.shape[0]):
            rowstoadd = pd.DataFrame({'Lat': final_table['Lat'],
                                      'Lon': final_table['Lon'],
                                      'PSC': final_table['PSC'],
                                      'SC_Avg_EcNo': final_table['SC_Avg_EcNo'],
                                      'SC_Avg_RSCP': final_table['SC_Avg_RSCP'],
                                      'MaxRSCP': final_table['MaxRSCP'],
                                      'MaxEcNo': final_table['MaxEcNo'],
                                      'count': final_table['count'],
                                      'PilotPollutionFlag': final_table['PilotPollutionFlag'],
                                      'LatCell': Cell1['Lat'],
                                      'LonCell': Cell1['Lon'],
                                      })
    Distance = Distance.append(rowstoadd, ignore_index=True)
    Distance['Min_Dist'] = CalcDistanceKM(Distance['LatCell'], Distance['LonCell'], Distance['Lat'], Distance['Lon'])
    Distance = Distance[
        ['Lat', 'Lon', 'PSC', 'SC_Avg_EcNo', 'SC_Avg_RSCP', 'MaxRSCP', 'MaxEcNo', 'count', 'PilotPollutionFlag',
         'Min_Dist']]
    Min_Dist = Distance.loc[Distance.groupby(['Lat', 'Lon'])['Min_Dist'].idxmin()].reset_index()
    # print (Min_Dist)
    Min_Dist.to_csv('final_file_pilot.csv')
    # Distance.to_csv('Distance.csv')

    pilot = pd.read_csv('final_data.csv')
    pilot_size = pilot.shape[0]
    Sum = pilot.groupby(['Lat', 'Lon'])
    Sum = Sum.count()
    Sum = Sum.rename(columns={'Unnamed: 0': 'SC_Cnt_final'})
    Sum = Sum[['SC_Cnt_final']].reset_index()
    final_table = pd.merge(final, Sum, how="left")
    final_table = final_table.drop(columns=['PSC', 'count', 'SC_Avg_EcNo', 'SC_Avg_RSCP', 'PilotPollutionFlag'])
    final_table = final_table.drop_duplicates()
    final_table.to_csv('finallll.csv')
    #
    pilot_map = pd.read_csv('finallll.csv')
    pilot_map_size = pilot_map.shape[0]
    Final_Report = pd.DataFrame(columns=['Lat', 'Lon', 'MaxEcNo', 'MaxRSCP', 'SC_Cnt_final', 'parcel_state'])
    for i in range(pilot_map_size):
        if pilot_map['SC_Cnt_final'][i] > 3:
            Final_Report = Final_Report.append({'Lat': pilot_map['Lat'][i],
                                                'Lon': pilot_map['Lon'][i],
                                                'MaxEcNo': pilot_map['MaxEcNo'][i],
                                                'MaxRSCP': pilot_map['MaxRSCP'][i],
                                                'SC_Cnt_final': pilot_map['SC_Cnt_final'][i],
                                                'parcel_state': 'pilot_problem'}, ignore_index=True)

        else:
            Final_Report = Final_Report.append({'Lat': pilot_map['Lat'][i],
                                                'Lon': pilot_map['Lon'][i],
                                                'MaxEcNo': pilot_map['MaxEcNo'][i],
                                                'MaxRSCP': pilot_map['MaxRSCP'][i],
                                                'SC_Cnt_final': pilot_map['SC_Cnt_final'][i],
                                                'parcel_state': 'no_pilot_problem'}, ignore_index=True)
    Final_Report = Final_Report.dropna()
    Final_Report.to_csv('final_data_pilot.csv')
    #

    ##plotting in google earth
    #pilot_Final = Final_Report[Final_Report['parcel_state'] == 'pilot_problem']
    #pilot_Final.to_csv('pilotproblem_Final.csv')
    #pilot_Final = pd.read_csv('pilotproblem_Final.csv')
    #pilot_Final_size = pilot_Final.shape[0]
    #no_pilot_Final = Final_Report[Final_Report['parcel_state'] == 'no_pilot_problem']
    #no_pilot_Final.to_csv('no_pilotproblem_Final.csv')
    #no_pilot_Final = pd.read_csv('no_pilotproblem_Final.csv')
    #no_pilot_Final_size = no_pilot_Final.shape[0]

#
# Start=time.process_time()
#
# sc= 'D:\GP\se.csv'
# x= 'D:\GP\A.csv'
#
# pilot(x,sc)
#
# End=time.process_time()
# Difference=End-Start
# print (Difference)
