# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 20:59:40 2020

@author: DELL
"""

import os
import pandas as pd
import numpy as np
import csv
from pandas import DataFrame
import groupbytime
from math import *
import csv


def CellState(cell, drive):
    Serving_Cell = pd.DataFrame()
    Missed_Data = pd.DataFrame()
    Final_Report = pd.DataFrame()
    Nearest_Cell = pd.DataFrame()

    Cell_file = pd.read_csv(cell)
    Drive_Test_File = pd.read_csv(drive)
    # Drive_Test_File = Drive_Test_File.iloc[ : 20 , :].reset_index()
    File_size = Drive_Test_File.shape[0]
    if File_size > 1000:
        step = 1000
        end = int(File_size / step) * step
        remaining = File_size % step
        for i in range(0, end, step):
            drive_test = Drive_Test_File.iloc[i: i + step, :].reset_index()  # get all data
            N, S, F, M = stateOpt(cell, drive_test)
            Nearest_Cell = Nearest_Cell.append(N, ignore_index=True)
            Serving_Cell = Serving_Cell.append(S, ignore_index=True)
            Final_Report = Final_Report.append(F, ignore_index=True)
            Missed_Data = Missed_Data.append(M, ignore_index=True)
        if remaining > 0:
            drive_test = Drive_Test_File.iloc[File_size - remaining:, :].reset_index()  # get all data
            N, S, F, M = stateOpt(cell, drive_test)
            Nearest_Cell = Nearest_Cell.append(N, ignore_index=True)
            Serving_Cell = Serving_Cell.append(S, ignore_index=True)
            Final_Report = Final_Report.append(F, ignore_index=True)
            Missed_Data = Missed_Data.append(M, ignore_index=True)
    else:
        drive_test = Drive_Test_File.iloc[:, :].reset_index()  # get all data
        N, S, F, M = stateOpt(cell, drive_test)
        Nearest_Cell = Nearest_Cell.append(N, ignore_index=True)
        Serving_Cell = Serving_Cell.append(S, ignore_index=True)
        Final_Report = Final_Report.append(F, ignore_index=True)
        Missed_Data = Missed_Data.append(M, ignore_index=True)
    Nearest_Cell.to_csv('Nearest_Cell.csv')
    Serving_Cell.to_csv('Serving_Cell.csv')
    Final_Report.to_csv('Final_Report.csv')
    Missed_Data.to_csv('Missed_Data.csv')
    Nearest_Final = Final_Report[Final_Report['States'] == 'Nearest']
    Nearest_Final.to_csv('Nearest_Final.csv')
    Nearest_Final = pd.read_csv('Nearest_Final.csv')
    Nearest_Final_size = Nearest_Final.shape[0]
    Diff_Sector = Final_Report[Final_Report['States'] == 'Diff_sector']
    Diff_Sector.to_csv('Diff_Sector.csv')
    Diff_Sector = pd.read_csv('Diff_Sector.csv')
    Diff_Sector_size = Diff_Sector.shape[0]
    Far_distance = Final_Report[Final_Report['States'] == 'serving cell is overshooting']
    Far_distance.to_csv('Far_distance.csv')
    Far_distance = pd.read_csv('Far_distance.csv')
    Far_distance_size = Far_distance.shape[0]
    Loaded = Final_Report[Final_Report['States'] == 'Loaded']
    Loaded.to_csv('Loaded.csv')
    Loaded = pd.read_csv('Loaded.csv')
    Loaded_size = Loaded.shape[0]
    Bad_Coverge = Final_Report[Final_Report['States'] == 'Bad Coverge']
    Bad_Coverge.to_csv('Bad Coverge.csv')
    Bad_Coverge = pd.read_csv('Bad Coverge.csv')
    Bad_Coverge_size = Bad_Coverge.shape[0]
    Blocked = Final_Report[Final_Report['States'] == 'Blocked']
    Blocked.to_csv('Blocked.csv')
    Blocked = pd.read_csv('Blocked.csv')
    Blocked_size = Blocked.shape[0]
    serving_cell_is_back_lobe = Final_Report[Final_Report['States'] == 'serving cell is back lobe']
    serving_cell_is_back_lobe.to_csv('serving_cell_is_back_lobe.csv')
    serving_cell_is_back_lobe = pd.read_csv('serving_cell_is_back_lobe.csv')
    serving_cell_is_back_lobe_size = serving_cell_is_back_lobe.shape[0]
    Final_Report.to_csv('Final_Report_test.csv', index=True)


"""
    import simplekml
    kml = simplekml.Kml()
    style = simplekml.Style()
    style.labelstyle.color = simplekml.Color.black  # Make the text red
    style.labelstyle.scale = 0.2  # Make the text twice as big
    style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
    style.iconstyle.color = 'ff008000'
    for i in range(Nearest_Final_size):
        lon = Nearest_Final['Lon_drive'][i]
        lat = Nearest_Final['Lat_drive'][i]
        coord = (lon, lat)
        pnt = kml.newpoint(name='nearest', coords=[(lon, lat)])
        pnt.style = style

    kml.save("nearest_final.kml")

    kml2 = simplekml.Kml()
    style2 = simplekml.Style()
    style2.labelstyle.color = simplekml.Color.black  # Make the text red
    style2.labelstyle.scale = 0.2  # Make the text twice as big
    style2.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
    style2.iconstyle.color = 'ff800080'
    for j in range(Diff_Sector_size):
        lon2 = Diff_Sector['Lon_drive'][j]
        lat2 = Diff_Sector['Lat_drive'][j]
        coord = (lon2, lat2)
        pnt_2 = kml2.newpoint(name='diff_sec', coords=[(lon2, lat2)])
        pnt_2.style = style2
    kml2.save("diff_sec_final.kml")

    kml3 = simplekml.Kml()
    style3 = simplekml.Style()
    style3.labelstyle.color = simplekml.Color.black  # Make the text red
    style3.labelstyle.scale = 0.2  # Make the text twice as big
    style3.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
    style3.iconstyle.color = 'ffff0000'
    for x in range(Far_distance_size):
        lon3 = Far_distance['Lon_drive'][x]
        lat3 = Far_distance['Lat_drive'][x]
        coord = (lon3, lat3)
        pnt_3 = kml3.newpoint(name='overshooting', coords=[(lon3, lat3)])
        pnt_3.style = style3

    kml3.save("overshooting.kml")

    kml4 = simplekml.Kml()
    style4 = simplekml.Style()
    style4.labelstyle.color = simplekml.Color.black  # Make the text red
    style4.labelstyle.scale = 0.2  # Make the text twice as big
    style4.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
    style4.iconstyle.color = 'ff00a5ff'
    for y in range(Loaded_size):
        lon4 = Loaded['Lon_drive'][y]
        lat4 = Loaded['Lat_drive'][y]
        coord = (lon4, lat4)
        pnt_4 = kml4.newpoint(name='Loaded', coords=[(lon4, lat4)])
        pnt_4.style = style4
    kml4.save("Loaded.kml")

    kml5 = simplekml.Kml()
    style5 = simplekml.Style()
    style5.labelstyle.color = simplekml.Color.black  # Make the text red
    style5.labelstyle.scale = 0.2  # Make the text twice as big
    style5.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
    style5.iconstyle.color = 'ff0000ff'
    for z in range(Blocked_size):
        lon5 = Blocked['Lon_drive'][z]
        lat5 = Blocked['Lat_drive'][z]
        coord = (lon5, lat5)
        pnt_5 = kml5.newpoint(name='Blocked', coords=[(lon5, lat5)])
        pnt_5.style = style5
    kml5.save("Blocked.kml")

    kml6 = simplekml.Kml()
    style6 = simplekml.Style()
    style6.labelstyle.color = simplekml.Color.black  # Make the text red
    style6.labelstyle.scale = 0.2  # Make the text twice as big
    style6.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
    style6.iconstyle.color = 'ff00ffff'
    for m in range(Bad_Coverge_size):
        lon6 = Bad_Coverge['Lon_drive'][m]
        lat6 = Bad_Coverge['Lat_drive'][m]
        coord = (lon6, lat6)
        pnt_6 = kml6.newpoint(name='Bad_Coverge', coords=[(lon6, lat6)])
        pnt_6.style = style6
    kml6.save("Bad_Coverge.kml")

    kml7 = simplekml.Kml()
    style7 = simplekml.Style()
    style7.labelstyle.color = simplekml.Color.black  # Make the text red
    style7.labelstyle.scale = 0.2  # Make the text twice as big
    style7.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
    style7.iconstyle.color = 'ffb469ff'
    for n in range(serving_cell_is_back_lobe_size):
        lon7 = serving_cell_is_back_lobe['Lon_drive'][n]
        lat7 = serving_cell_is_back_lobe['Lat_drive'][n]
        coord = (lon7, lat7)
        pnt_7 = kml7.newpoint(name='serving_cell_is_back_lobe', coords=[(lon7, lat7)])
        pnt_7.style = style7
    kml7.save("serving_cell_is_back_lobe.kml")
"""


#    Missed_Data.to_csv('Missed_Data_test.csv', index=True)
#    Final_Report.to_csv('Final_Report_test.csv', index=True)
# print(Final_Report)

def calcBearing(a_lat, a_lng, b_lat, b_lng):
    a_lat = np.radians(a_lat)
    a_lng = np.radians(a_lng)
    b_lat = np.radians(b_lat)
    b_lng = np.radians(b_lng)
    d_lng = b_lng - a_lng
    y = np.cos(b_lat) * np.sin(d_lng)
    x = np.cos(a_lat) * np.sin(b_lat) - np.sin(a_lat) * np.cos(b_lat) * np.cos(d_lng)
    Bearing = np.arctan2(y, x)
    Bearing = np.degrees(Bearing)
    Bearing = (Bearing + 360) % 360
    return Bearing


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


def stateOpt(x, Drive_Test_File):
    Cell_file = pd.read_csv(x)
    # Drive_Test_File = pd.read_csv(y)
    Cells = Cell_file[['Region', 'Cell', 'Lat', 'Lon', 'SC', 'UARFCN', 'ANT_DIRECTION']]
    drive_test = Drive_Test_File[
        ['Latitude', 'Longitude', 'Categorized PSC:A1', 'Categorized UARFCN_DL:A1', 'Categorized PSC:A2',
         'Categorized UARFCN_DL:A2', 'Categorized PSC:A3', 'Categorized UARFCN_DL:A3',
         'Categorized Ec/Io:A1', 'Categorized Ec/Io:A2', 'Categorized Ec/Io:A3', 'Categorized RSCP:A1',
         'Categorized RSCP:A2', 'Categorized RSCP:A3']]

    # drive_test = drive_test.iloc[: , :].reset_index() #get all data
    Serving_Cell = pd.DataFrame(columns=['ref_Index', 'Lat_Cell', 'Lon_Cell', 'Serving_Cell', 'Dis_Serving',
                                         'Bearing', 'ANT_DIRECTION', 'ECNo_A1', 'RSCP_A1', 'drive_lat', 'drive_lon'])
    Missed_Data = pd.DataFrame(columns=['Lat', 'Lon', 'ref_Index', 'States', 'SC', 'UARFCN'])
    Final_Report = pd.DataFrame(
        columns=['Lat_drive', 'Lon_drive', 'Lat_Cell', 'Lon_Cell', 'Ant_Dirction', 'Cell', 'States'])
    Nearest_Cell = pd.DataFrame(columns=['ref_Index', 'Dis_Nearest', 'ANT_DIRECTION', 'Lat_Cell', 'Lat_drive',
                                         'Lon_Cell', 'Lat_drive', 'Nearest', 'SC', 'UARFCN', '2nd_Nearest',
                                         'Dis_2nd_Nrst', 'Reference_Dis'])
    # Drop NAN columns#
    drive_test = drive_test[np.isfinite(drive_test['Latitude'])]
    drive_test = drive_test[np.isfinite(drive_test['Longitude'])]
    drive_test = drive_test.reset_index()

    # get data
    Sus_Nearest_Cell = pd.DataFrame(columns=['ref_Index', 'Dis_Nearest'])
    Serving_Cell = pd.DataFrame(columns=['ref_Index', 'Lat_Cell', 'Lon_Cell', 'Serving_Cell', 'Dis_Serving',
                                         'Bearing', 'ANT_DIRECTION', 'ECNo_A1', 'RSCP_A1', 'drive_lat', 'drive_lon'])
    A2 = pd.DataFrame(
        columns=['ref_Index', 'A2', 'ECNo_A2', 'RSCP_A2', 'Distance_A2', 'drive_lat', 'drive_lon', 'Lat_Cell',
                 'Lon_Cell'])
    A3 = pd.DataFrame(
        columns=['ref_Index', 'A3', 'ECNo_A3', 'RSCP_A3', 'Distance_A3', 'drive_lat', 'drive_lon', 'Lat_Cell',
                 'Lon_Cell'])
    Final_Report = pd.DataFrame(
        columns=['Lat_drive', 'Lon_drive', 'Lat_Cell', 'Lon_Cell', 'Ant_Dirction', 'Cell', 'States'])
    drive_test_size = drive_test.shape[0]
    Cells_size = Cells.shape[0]
    Missed_Data = pd.DataFrame(columns=['Lat', 'Lon', 'ref_Index', 'States', 'SC', 'UARFCN'])
    X1 = 14 / 100
    X2 = 5
    X3 = -100
    X4 = 10
    # Region filter
    cell = Cells[['Region', 'Lat', 'Lon']]
    drive1 = pd.DataFrame(columns=['Latitude', 'Longitude'])
    drive1 = Drive_Test_File[['Latitude', 'Longitude']]
    drive1 = drive1.iloc[:200, :]
    Cells_size = cell.shape[0]
    drive1 = drive1.iloc[np.full(Cells_size, 0)].reset_index()
    cell['region'] = np.where(
        (abs(cell['Lat'] - drive1['Latitude']) < 0.1) & (abs(cell['Lon'] - drive1['Longitude']) < 0.1), 'True', 'False')
    cell = cell[cell['region'] == 'True']
    cell_region = cell.iloc[0, 0]
    Cells = Cells[Cells['Region'] == cell_region].reset_index()
    Cells_size = Cells.shape[0]
    ##
    Cells1 = pd.DataFrame(columns=['Cell', 'Lat', 'Lon', 'SC', 'UARFCN', 'ANT_DIRECTION'])

    #
    for i in range(0, drive_test.shape[0], 1):
        drive_sample = DataFrame({'drive_lat': drive_test['Latitude'][i],
                                  'drive_lon': drive_test['Longitude'][i],
                                  'A1': drive_test['Categorized PSC:A1'][i],
                                  'UARFCN_A1': drive_test['Categorized UARFCN_DL:A1'][i],
                                  'A2': drive_test['Categorized PSC:A2'][i],
                                  'UARFCN_A2': drive_test['Categorized UARFCN_DL:A2'][i],
                                  'A3': drive_test['Categorized PSC:A3'][i],
                                  'UARFCN_A3': drive_test['Categorized UARFCN_DL:A3'][i],
                                  'Ec/Io:A1': drive_test['Categorized Ec/Io:A1'][i],
                                  'Ec/Io:A2': drive_test['Categorized Ec/Io:A2'][i],
                                  'Ec/Io:A3': drive_test['Categorized Ec/Io:A3'][i],
                                  'RSCP:A1': drive_test['Categorized RSCP:A1'][i],
                                  'RSCP:A2': drive_test['Categorized RSCP:A2'][i],
                                  'RSCP:A3': drive_test['Categorized RSCP:A3'][i]},
                                 columns=['drive_lat', 'drive_lon', 'A1', 'UARFCN_A1', 'A2', 'UARFCN_A2', 'A3',
                                          'UARFCN_A3', 'Ec/Io:A1', 'Ec/Io:A2', 'Ec/Io:A3', 'RSCP:A1', 'RSCP:A2',
                                          'RSCP:A3'], index=[0])
        drive_sample = drive_sample.iloc[np.full(Cells_size, 0)]
        drive_sample = drive_sample.reset_index()
        Cells1 = Cells
        Cells2 = Cells
        Cells_F1 = Cells
        Cells_A2 = Cells
        Cells_A3 = Cells
        # get frist range
        Cells_F1['Cond1_f1'] = np.where((abs(drive_sample['drive_lat'] - Cells['Lat']) < 0.1) & (
                    abs(drive_sample['drive_lon'] - Cells['Lon']) < 0.1), 'True', 'False')
        Cells_F1 = Cells_F1[Cells_F1['Cond1_f1'] == "True"].reset_index()
        # get Serving_Cell
        Cells1['Cond1_A1'] = np.where((abs(drive_sample['drive_lat'] - Cells['Lat']) < 0.1) & (
                    abs(drive_sample['drive_lon'] - Cells['Lon']) < 0.1) & (drive_sample['A1'] == Cells['SC']) & (
                                                  drive_sample['UARFCN_A1'] == Cells['UARFCN']), 'True', 'False')
        Cells1 = Cells1[Cells1['Cond1_A1'] == "True"].reset_index()
        if Cells1.shape[0] > 0:
            Serving_Cell = Serving_Cell.append({'ref_Index': int(i) - 1, 'Lat_Cell': Cells1['Lat'][0],
                                                'Lon_Cell': Cells1['Lon'][0],
                                                'ANT_DIRECTION': Cells1['ANT_DIRECTION'][0],
                                                'Serving_Cell': Cells1['Cell'][0],
                                                'ECNo_A1': drive_sample['Ec/Io:A1'][0],
                                                'RSCP_A1': drive_sample['RSCP:A1'][0],
                                                'drive_lat': drive_sample['drive_lat'][0],
                                                'drive_lon': drive_sample['drive_lon'][0]},
                                               ignore_index=True)
        # get A2
        Cells_A2['Cond1_A2'] = np.where((abs(drive_sample['drive_lat'] - Cells['Lat']) < 0.1) & (
                    abs(drive_sample['drive_lon'] - Cells['Lon']) < 0.1) & (drive_sample['A2'] == Cells['SC']) & (
                                                    drive_sample['UARFCN_A2'] == Cells['UARFCN']), 'True', 'False')
        Cells_A2 = Cells_A2[Cells_A2['Cond1_A2'] == "True"].reset_index()
        if Cells_A2.shape[0] > 0:
            A2 = A2.append({'drive_lat': drive_sample['drive_lat'][0],
                            'drive_lon': drive_sample['drive_lon'][0],
                            'Lat_Cell': Cells_A2['Lat'][0],
                            'Lon_Cell': Cells_A2['Lon'][0],
                            'ref_Index': int(i) - 1, 'A2': Cells_A2['Cell'][0], 'ECNo_A2': drive_sample['Ec/Io:A2'][0],
                            'RSCP_A2': drive_sample['RSCP:A2'][0]},
                           ignore_index=True)
        # get A3
        Cells_A3['Cond1_A3'] = np.where((abs(drive_sample['drive_lat'] - Cells['Lat']) < 0.1) & (
                    abs(drive_sample['drive_lon'] - Cells['Lon']) < 0.1) & (drive_sample['A3'] == Cells['SC']) & (
                                                    drive_sample['UARFCN_A3'] == Cells['UARFCN']), 'True', 'False')
        Cells_A3 = Cells_A3[Cells_A3['Cond1_A3'] == "True"].reset_index()
        if Cells_A3.shape[0] > 0:
            A3 = A3.append({'drive_lat': drive_sample['drive_lat'][0],
                            'drive_lon': drive_sample['drive_lon'][0],
                            'Lat_Cell': Cells_A3['Lat'][0],
                            'Lon_Cell': Cells_A3['Lon'][0],
                            'ref_Index': int(i) - 1, 'A3': Cells_A3['Cell'][0], 'ECNo_A3': drive_sample['Ec/Io:A3'][0],
                            'RSCP_A3': drive_sample['RSCP:A3'][0]},
                           ignore_index=True)
        drive_sample2 = DataFrame({'drive_lat': drive_test['Latitude'][i],
                                   'drive_lon': drive_test['Longitude'][i],
                                   'A1': drive_test['Categorized PSC:A1'][i],
                                   'UARFCN_A1': drive_test['Categorized UARFCN_DL:A1'][i]},
                                  columns=['drive_lat', 'drive_lon', 'A1', 'UARFCN_A1'], index=[0])
        drive_sample2 = drive_sample2.iloc[np.full(Cells_F1.shape[0], 0)]
        drive_sample2 = drive_sample2.reset_index()
        Cells_F1['Bearing'] = calcBearing(Cells_F1['Lat'], Cells_F1['Lon'], drive_sample2['drive_lat'],
                                          drive_sample2['drive_lon'])
        Cells2 = Cells_F1
        Cells2['Cond2'] = np.where((abs(drive_sample2['drive_lat'] - Cells_F1['Lat']) < 0.026) & (
                    abs(drive_sample2['drive_lon'] - Cells_F1['Lon']) < 0.026), 'True', 'False')
        Cells2 = Cells2[Cells2['Cond2'] == "True"]
        Cells2 = Cells2.reset_index(drop=True)
        Cells2['Border1'] = Cells2['ANT_DIRECTION'] + 30
        Cells2['Border2'] = Cells2['ANT_DIRECTION'] - 30
        drive_sample2 = drive_sample2.iloc[:Cells2.shape[0], :]
        Cells2['Bearing1'] = np.where((Cells2['Border1'] > 360) & (Cells2['Bearing'] < 60), Cells2['Bearing'] + 360,
                                      Cells2['Bearing'])
        Cells2 = Cells2.reset_index(drop=True)
        Cells3 = Cells2
        Cells3['cond'] = np.where(
            (Cells2['Bearing1'] <= Cells2['Border1']) & (Cells2['Bearing1'] >= Cells2['Border2']) & (
                        drive_sample2['UARFCN_A1'] == Cells2['UARFCN']), 'True', 'False')
        Cells3 = Cells3[Cells3['cond'] == "True"]
        Cells3 = Cells3.reset_index(drop=True)
        if Cells3.shape[0] > 0:
            for j in range(0, Cells3.shape[0], 1):
                Sus_Nearest_Cell = Sus_Nearest_Cell.append({'Lat_drive': drive_sample2['drive_lat'][0],
                                                            'Lon_drive': drive_sample2['drive_lon'][0],
                                                            'Lat_Cell': Cells3['Lat'][j],
                                                            'Lon_Cell': Cells3['Lon'][j],
                                                            'ref_Index': int(i) - 1,
                                                            'ANT_DIRECTION': Cells3['ANT_DIRECTION'][j],
                                                            'SC': drive_sample2['A1'][0],
                                                            'UARFCN': drive_sample2['UARFCN_A1'][0],
                                                            'Bearing': Cells3['Bearing1'][j],
                                                            'Nearest': Cells3['Cell'][j]}, ignore_index=True)

    # Calculting distance & Bearing
    Serving_Cell['Dis_Serving'] = CalcDistanceKM(Serving_Cell['Lat_Cell'], Serving_Cell['Lon_Cell'],
                                                 Serving_Cell['drive_lat'], Serving_Cell['drive_lon'])
    A2['Distance_A2'] = CalcDistanceKM(A2['Lat_Cell'], A2['Lon_Cell'], A2['drive_lat'], A2['drive_lon'])
    A3['Distance_A3'] = CalcDistanceKM(A3['Lat_Cell'], A3['Lon_Cell'], A3['drive_lat'], A3['drive_lon'])
    A2 = A2.iloc[:, :5]
    A3 = A3.iloc[:, :5]
    Serving_Cell['Bearing'] = calcBearing(Serving_Cell['Lat_Cell'], Serving_Cell['Lon_Cell'], Serving_Cell['drive_lat'],
                                          Serving_Cell['drive_lon'])
    Sus_Nearest_Cell['Dis_Nearest'] = CalcDistanceKM(Sus_Nearest_Cell['Lat_Cell'], Sus_Nearest_Cell['Lon_Cell'],
                                                     Sus_Nearest_Cell['Lat_drive'], Sus_Nearest_Cell['Lon_drive'])

    Serving_Cell = Serving_Cell.loc[Serving_Cell.groupby(['ref_Index'])['Dis_Serving'].idxmin()].reset_index()
    A2 = A2.loc[A2.groupby(['ref_Index'])['Distance_A2'].idxmin()].reset_index()
    A3 = A3.loc[A3.groupby(['ref_Index'])['Distance_A3'].idxmin()].reset_index()
    Serving_Cell = pd.merge(Serving_Cell, A2, on="ref_Index", how="left")
    Serving_Cell = pd.merge(Serving_Cell, A3, on="ref_Index", how="left")
    Serving_Cell = Serving_Cell[['ref_Index', 'Lat_Cell', 'Lon_Cell', 'Serving_Cell', 'Dis_Serving',
                                 'Bearing', 'ANT_DIRECTION', 'ECNo_A1', 'RSCP_A1', 'A2', 'ECNo_A2',
                                 'RSCP_A2', 'A3', 'ECNo_A3', 'RSCP_A3']]

    # Get Nearest Cells
    Nearest_Cell = Sus_Nearest_Cell.loc[Sus_Nearest_Cell.groupby(['ref_Index'])['Dis_Nearest'].idxmin()]
    N = Nearest_Cell[['Dis_Nearest']]
    N = N.rename(columns={"Dis_Nearest": "N"})
    N = pd.concat([Sus_Nearest_Cell, N], axis=1, sort=False)
    Nearest_Cell = Nearest_Cell.reset_index()
    N['N'] = np.where(N['N'] == N['Dis_Nearest'], 'true', 'false')
    Sus_Nearest_Cell = N[N['N'] == "false"]
    Sus_Nearest_Cell = Sus_Nearest_Cell.iloc[:, :10].reset_index()
    second_Nearest = Sus_Nearest_Cell.loc[Sus_Nearest_Cell.groupby(['ref_Index'])['Dis_Nearest'].idxmin()].reset_index()
    second_Nearest = second_Nearest.rename(columns={'Dis_Nearest': 'Dis_2nd_Nrst', 'Nearest': '2nd_Nearest'})
    second_Nearest = second_Nearest[['ref_Index', '2nd_Nearest', 'Dis_2nd_Nrst']]
    Nearest_Cell = pd.merge(Nearest_Cell, second_Nearest, on="ref_Index", how="left")

    # Nearest_Cell.to_csv('Nearest_Final1.csv')
    # Serving_Cell.to_csv('Serving_Final1.csv')

    Nearest_Cell['Reference_Dis'] = Nearest_Cell['Dis_Nearest'] + X1 * Nearest_Cell['Dis_Nearest']
    for i in range(0, Nearest_Cell.shape[0], 1):
        Nearest_sample = DataFrame({'ref_Index': Nearest_Cell['ref_Index'][i],
                                    'Dis_Nearest': Nearest_Cell['Dis_Nearest'][i],
                                    'ANT_DIRECTION': Nearest_Cell['ANT_DIRECTION'][i],
                                    'Bearing': Nearest_Cell['Bearing'][i],
                                    'Lat_Cell': Nearest_Cell['Lat_Cell'][i],
                                    'Lat_drive': Nearest_Cell['Lat_drive'][i],
                                    'Lon_Cell': Nearest_Cell['Lon_Cell'][i],
                                    'Lon_drive': Nearest_Cell['Lon_drive'][i],
                                    'Nearest': Nearest_Cell['Nearest'][i],
                                    'SC': Nearest_Cell['SC'][i],
                                    'UARFCN': Nearest_Cell['UARFCN'][i],
                                    '2nd_Nearest': Nearest_Cell['2nd_Nearest'][i],
                                    'Dis_2nd_Nrst': Nearest_Cell['Dis_2nd_Nrst'][i],
                                    'Reference_Dis': Nearest_Cell['Reference_Dis'][i]},
                                   columns=['ref_Index', 'Dis_Nearest', 'ANT_DIRECTION', 'Bearing', 'Lat_Cell',
                                            'Lat_drive',
                                            'Lon_Cell', 'Lon_drive', 'Nearest', 'SC', 'UARFCN', '2nd_Nearest',
                                            'Dis_2nd_Nrst', 'Reference_Dis'], index=[0])
        Nearest_sample = Nearest_sample.iloc[np.full(Serving_Cell.shape[0], 0)]
        Nearest_sample = Nearest_sample.reset_index()
        Serv1 = Serving_Cell
        Serv1['c'] = np.where((Nearest_sample['ref_Index'] == Serving_Cell['ref_Index']), 'True', 'False')
        Serv1 = Serv1[Serv1['c'] == "True"].reset_index()
        if Serv1.shape[0] > 0:
            if (Nearest_sample['Nearest'][0] == Serv1['Serving_Cell'][0]) or (
                    Serv1['Dis_Serving'][0] <= Nearest_sample['Reference_Dis'][0]):
                Final_Report = Final_Report.append({'Lat_drive': Nearest_Cell['Lat_drive'][i],
                                                    'Lon_drive': Nearest_Cell['Lon_drive'][i],
                                                    'Lat_Cell': Serv1['Lat_Cell'][0],
                                                    'Lon_Cell': Serv1['Lon_Cell'][0],
                                                    'Cell': Serv1['Serving_Cell'][0],
                                                    'Ant_Dirction': Serv1['ANT_DIRECTION'][0],
                                                    'States': 'Nearest'}, ignore_index=True)

            elif Nearest_sample['Dis_Nearest'][0] == Serv1['Dis_Serving'][0]:
                Final_Report = Final_Report.append({'Lat_drive': Nearest_Cell['Lat_drive'][i],
                                                    'Lon_drive': Nearest_Cell['Lon_drive'][i],
                                                    'Lat_Cell': Serv1['Lat_Cell'][0],
                                                    'Lon_Cell': Serv1['Lon_Cell'][0],
                                                    'Cell': Serv1['Serving_Cell'][0],
                                                    'Ant_Dirction': Serv1['ANT_DIRECTION'][0],
                                                    'States': 'Diff_sector'}, ignore_index=True)

            elif Serv1['Dis_Serving'][0] > Nearest_sample['Dis_Nearest'][0] + Nearest_sample['Dis_2nd_Nrst'][0]:
                Final_Report = Final_Report.append({'Lat_drive': Nearest_Cell['Lat_drive'][i],
                                                    'Lon_drive': Nearest_Cell['Lon_drive'][i],
                                                    'Lat_Cell': Serv1['Lat_Cell'][0],
                                                    'Lon_Cell': Serv1['Lon_Cell'][0],
                                                    'Cell': Serv1['Serving_Cell'][0],
                                                    'Ant_Dirction': Serv1['ANT_DIRECTION'][0],
                                                    'States': 'serving cell is overshooting'}, ignore_index=True)

            elif Serv1['Dis_Serving'][0] < Nearest_sample['Dis_Nearest'][0] + Nearest_sample['Dis_2nd_Nrst'][0]:
                if Nearest_sample['Nearest'][0] == Serv1['A2'][0]:
                    if Serv1['ECNo_A1'][0] - Serv1['ECNo_A2'][0] > X2:
                        Final_Report = Final_Report.append({'Lat_drive': Nearest_Cell['Lat_drive'][i],
                                                            'Lon_drive': Nearest_Cell['Lon_drive'][i],
                                                            'Lat_Cell': Nearest_sample['Lat_Cell'][0],
                                                            'Lon_Cell': Nearest_sample['Lon_Cell'][0],
                                                            'Cell': Nearest_sample['Nearest'][0],
                                                            'Ant_Dirction': Nearest_sample['ANT_DIRECTION'][0],
                                                            'States': 'Loaded'}, ignore_index=True)

                    elif Serv1['ECNo_A1'][0] - Serv1['ECNo_A2'][0] < X2:
                        if Serv1['RSCP_A1'][0] < X3:
                            Final_Report = Final_Report.append({'Lat_drive': Nearest_Cell['Lat_drive'][i],
                                                                'Lon_drive': Nearest_Cell['Lon_drive'][i],
                                                                'Lat_Cell': Serv1['Lat_Cell'][0],
                                                                'Lon_Cell': Serv1['Lon_Cell'][0],
                                                                'Cell': Serv1['Serving_Cell'][0],
                                                                'Ant_Dirction': Serv1['ANT_DIRECTION'][0],
                                                                'States': 'Bad Coverge'}, ignore_index=True)

                        elif Serv1['RSCP_A1'][0] - Serv1['RSCP_A2'][0] > X4:
                            Final_Report = Final_Report.append({'Lat_drive': Nearest_Cell['Lat_drive'][i],
                                                                'Lon_drive': Nearest_Cell['Lon_drive'][i],
                                                                'Lat_Cell': Nearest_sample['Lat_Cell'][0],
                                                                'Lon_Cell': Nearest_sample['Lon_Cell'][0],
                                                                'Cell': Nearest_sample['Nearest'][0],
                                                                'Ant_Dirction': Nearest_sample['ANT_DIRECTION'][0],
                                                                'States': 'Blocked'}, ignore_index=True)

                        elif Serv1['RSCP_A1'][0] - Serv1['RSCP_A2'][0] < X4:
                            Final_Report = Final_Report.append({'Lat_drive': Nearest_Cell['Lat_drive'][i],
                                                                'Lon_drive': Nearest_Cell['Lon_drive'][i],
                                                                'Lat_Cell': Serv1['Lat_Cell'][0],
                                                                'Lon_Cell': Serv1['Lon_Cell'][0],
                                                                'Cell': Serv1['Serving_Cell'][0],
                                                                'Ant_Dirction': Serv1['ANT_DIRECTION'][0],
                                                                'States': 'Nearest'}, ignore_index=True)

                elif Nearest_sample['Nearest'][0] == Serv1['A3'][0]:
                    if Serv1['ECNo_A1'][0] - Serv1['ECNo_A3'][0] > X2:
                        Final_Report = Final_Report.append({'Lat_drive': Nearest_Cell['Lat_drive'][i],
                                                            'Lon_drive': Nearest_Cell['Lon_drive'][i],
                                                            'Lat_Cell': Nearest_sample['Lat_Cell'][0],
                                                            'Lon_Cell': Nearest_sample['Lon_Cell'][0],
                                                            'Cell': Nearest_sample['Nearest'][0],
                                                            'Ant_Dirction': Nearest_sample['ANT_DIRECTION'][0],
                                                            'States': 'Loaded'}, ignore_index=True)

                    elif Serv1['ECNo_A1'][0] - Serv1['ECNo_A3'][0] < X2:
                        if Serv1['RSCP_A1'][0] < X3:
                            Final_Report = Final_Report.append({'Lat_drive': Nearest_Cell['Lat_drive'][i],
                                                                'Lon_drive': Nearest_Cell['Lon_drive'][i],
                                                                'Lat_Cell': Serv1['Lat_Cell'][0],
                                                                'Lon_Cell': Serv1['Lon_Cell'][0],
                                                                'Cell': Serv1['Serving_Cell'][0],
                                                                'Ant_Dirction': Serv1['ANT_DIRECTION'][0],
                                                                'States': 'Bad Coverge'}, ignore_index=True)

                        elif Serv1['RSCP_A1'][0] - Serv1['RSCP_A3'][0] > X4:
                            Final_Report = Final_Report.append({'Lat_drive': Nearest_Cell['Lat_drive'][i],
                                                                'Lon_drive': Nearest_Cell['Lon_drive'][i],
                                                                'Lat_Cell': Nearest_sample['Lat_Cell'][0],
                                                                'Lon_Cell': Nearest_sample['Lon_Cell'][0],
                                                                'Cell': Nearest_sample['Nearest'][0],
                                                                'Ant_Dirction': Nearest_sample['ANT_DIRECTION'][0],
                                                                'States': 'Blocked'}, ignore_index=True)

                        elif Serv1['RSCP_A1'][0] - Serv1['RSCP_A3'][0] < X4:
                            Final_Report = Final_Report.append({'Lat_drive': Nearest_Cell['Lat_drive'][i],
                                                                'Lon_drive': Nearest_Cell['Lon_drive'][i],
                                                                'Lat_Cell': Nearest_sample['Lat_Cell'][0],
                                                                'Lon_Cell': Nearest_sample['Lon_Cell'][0],
                                                                'Cell': Nearest_sample['Nearest'][0],
                                                                'Ant_Dirction': Nearest_sample['ANT_DIRECTION'][0],
                                                                'States': 'Nearest'}, ignore_index=True)
                else:
                    Final_Report = Final_Report.append({'Lat_drive': Nearest_Cell['Lat_drive'][i],
                                                        'Lon_drive': Nearest_Cell['Lon_drive'][i],
                                                        'Lat_Cell': Nearest_sample['Lat_Cell'][0],
                                                        'Lon_Cell': Nearest_sample['Lon_Cell'][0],
                                                        'Cell': Nearest_sample['Nearest'][0],
                                                        'Ant_Dirction': Nearest_sample['ANT_DIRECTION'][0],
                                                        'States': 'Blocked'}, ignore_index=True)

            elif Serv1['Dis_Serving'][0] < Nearest_sample['Dis_Nearest'][0] and Serv1['Dis_Serving'][
                j] < 0.1:
                Final_Report = Final_Report.append({'Lat_drive': Nearest_Cell['Lat_drive'][i],
                                                    'Lon_drive': Nearest_Cell['Lon_drive'][i],
                                                    'Lat_Cell': Serv1['Lat_Cell'][0],
                                                    'Lon_Cell': Serv1['Lon_Cell'][0],
                                                    'Cell': Serv1['Serving_Cell'][0],
                                                    'Ant_Dirction': Serv1['ANT_DIRECTION'][0],
                                                    'States': 'serving cell is back lobe'}, ignore_index=True)

            elif Serv1['Dis_Serving'][0] < Nearest_sample['Dis_Nearest'][0] and Serv1['Dis_Serving'][
                j] > 0.1:
                Final_Report = Final_Report.append({'Lat_drive': Nearest_Cell['Lat_drive'][i],
                                                    'Lon_drive': Nearest_Cell['Lon_drive'][i],
                                                    'Lat_Cell': Serv1['Lat_Cell'][0],
                                                    'Lon_Cell': Serv1['Lon_Cell'][0],
                                                    'Cell': Serv1['Serving_Cell'][0],
                                                    'Ant_Dirction': Serv1['ANT_DIRECTION'][0],
                                                    'States': 'Diff_sector'},
                                                   ignore_index=True)
        else:
            Missed_Data = Missed_Data.append({'Lat': Nearest_Cell['Lat_drive'][i],
                                              'Lon': Nearest_Cell['Lon_drive'][i],
                                              'ref_Index': Nearest_Cell['ref_Index'][i],
                                              'SC': Nearest_Cell['SC'][i],
                                              'UARFCN': Nearest_Cell['UARFCN'][i],
                                              'States': 'Missed_Data'}, ignore_index=True)
    Nearest_Cell = Nearest_Cell[['ref_Index', 'Dis_Nearest', 'ANT_DIRECTION', 'Bearing', 'Lat_Cell', 'Lat_drive',
                                 'Lon_Cell', 'Lon_drive', 'Nearest', 'SC', 'UARFCN', '2nd_Nearest',
                                 'Dis_2nd_Nrst']]
    return Nearest_Cell, Serving_Cell, Final_Report, Missed_Data
