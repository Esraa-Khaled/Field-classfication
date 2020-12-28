import pandas as pd
from pandas import DataFrame
import numpy as np
# import groupbytime
from math import *


def Missing(cell, drive):
    Drive_Test_File = pd.read_csv(drive)
    # Drive_Test_File = Drive_Test_File.iloc[ : 200 , :].reset_index()
    File_size = Drive_Test_File.shape[0]
    if File_size > 300:
        step = 300
        end = int(File_size / step) * step
        remaining = File_size % step
        for i in range(0, end, step):
            drive_test = Drive_Test_File.iloc[i: i + step, :].reset_index()  # get all data
            A1, A2, A3, D1, D2, D3, D4, D5, D6, M1, M2, M3, M4, M5, M6 = calc(cell, drive_test)
        if remaining > 0:
            drive_test = Drive_Test_File.iloc[File_size - remaining:, :].reset_index()  # get all data
            A1, A2, A3, D1, D2, D3, D4, D5, D6, M1, M2, M3, M4, M5, M6 = calc(cell, drive_test)
    else:
        drive_test = Drive_Test_File.iloc[:, :].reset_index()  # get all data
        A1, A2, A3, D1, D2, D3, D4, D5, D6, M1, M2, M3, M4, M5, M6 = calc(cell, drive_test)

    A1['Distance'] = CalcDistanceKM(A1['Lat_Cell'], A1['Lon_Cell'], A1['drive_lat'], A1['drive_lon'])
    A2['Distance'] = CalcDistanceKM(A2['Lat_Cell'], A2['Lon_Cell'], A2['drive_lat'], A2['drive_lon'])
    A3['Distance'] = CalcDistanceKM(A3['Lat_Cell'], A3['Lon_Cell'], A3['drive_lat'], A3['drive_lon'])
    D1['Distance'] = CalcDistanceKM(D1['Lat_Cell'], D1['Lon_Cell'], D1['drive_lat'], D1['drive_lon'])
    D2['Distance'] = CalcDistanceKM(D2['Lat_Cell'], D2['Lon_Cell'], D2['drive_lat'], D2['drive_lon'])
    D3['Distance'] = CalcDistanceKM(D3['Lat_Cell'], D3['Lon_Cell'], D3['drive_lat'], D3['drive_lon'])
    D4['Distance'] = CalcDistanceKM(D4['Lat_Cell'], D4['Lon_Cell'], D4['drive_lat'], D4['drive_lon'])
    D5['Distance'] = CalcDistanceKM(D5['Lat_Cell'], D5['Lon_Cell'], D5['drive_lat'], D5['drive_lon'])
    D6['Distance'] = CalcDistanceKM(D6['Lat_Cell'], D6['Lon_Cell'], D6['drive_lat'], D6['drive_lon'])
    M1['Distance'] = CalcDistanceKM(M1['Lat_Cell'], M1['Lon_Cell'], M1['drive_lat'], M1['drive_lon'])
    M2['Distance'] = CalcDistanceKM(M2['Lat_Cell'], M2['Lon_Cell'], M2['drive_lat'], M2['drive_lon'])
    M3['Distance'] = CalcDistanceKM(M3['Lat_Cell'], M3['Lon_Cell'], M3['drive_lat'], M3['drive_lon'])
    M4['Distance'] = CalcDistanceKM(M4['Lat_Cell'], M4['Lon_Cell'], M4['drive_lat'], M4['drive_lon'])
    M5['Distance'] = CalcDistanceKM(M5['Lat_Cell'], M5['Lon_Cell'], M5['drive_lat'], M5['drive_lon'])
    M6['Distance'] = CalcDistanceKM(M6['Lat_Cell'], M6['Lon_Cell'], M6['drive_lat'], M6['drive_lon'])

    # filter the lists of A, M & D and getting the index that has the min distance
    A1 = A1.loc[A1.groupby(['ref_Index'])['Distance'].idxmin()].reset_index()
    A2 = A2.loc[A2.groupby(['ref_Index'])['Distance'].idxmin()].reset_index()
    A3 = A3.loc[A3.groupby(['ref_Index'])['Distance'].idxmin()].reset_index()
    M1 = M1.loc[M1.groupby(['ref_Index'])['Distance'].idxmin()].reset_index()
    M2 = M2.loc[M2.groupby(['ref_Index'])['Distance'].idxmin()].reset_index()
    M3 = M3.loc[M3.groupby(['ref_Index'])['Distance'].idxmin()].reset_index()
    M4 = M4.loc[M4.groupby(['ref_Index'])['Distance'].idxmin()].reset_index()
    M5 = M5.loc[M5.groupby(['ref_Index'])['Distance'].idxmin()].reset_index()
    M6 = M6.loc[M6.groupby(['ref_Index'])['Distance'].idxmin()].reset_index()
    D1 = D1.loc[D1.groupby(['ref_Index'])['Distance'].idxmin()].reset_index()
    D2 = D2.loc[D2.groupby(['ref_Index'])['Distance'].idxmin()].reset_index()
    D3 = D3.loc[D3.groupby(['ref_Index'])['Distance'].idxmin()].reset_index()
    D4 = D4.loc[D4.groupby(['ref_Index'])['Distance'].idxmin()].reset_index()
    D5 = D5.loc[D5.groupby(['ref_Index'])['Distance'].idxmin()].reset_index()
    D6 = D6.loc[D6.groupby(['ref_Index'])['Distance'].idxmin()].reset_index()

    # "merge=concat", merging A1 to all cells
    # on= ref_Index, coloum on which I want to perfom merge
    # how= inner (default value), take the intersections only"coloums which have the same ref_Index"
    A1_A2 = pd.merge(A2, A1, on="ref_Index", how="inner")
    A1_A3 = pd.merge(A3, A1, on="ref_Index", how="inner")
    A1_M1 = pd.merge(M1, A1, on="ref_Index", how="inner")
    A1_M2 = pd.merge(M2, A1, on="ref_Index", how="inner")
    A1_M3 = pd.merge(M3, A1, on="ref_Index", how="inner")
    A1_M4 = pd.merge(M4, A1, on="ref_Index", how="inner")
    A1_M5 = pd.merge(M5, A1, on="ref_Index", how="inner")
    A1_M6 = pd.merge(M6, A1, on="ref_Index", how="inner")
    A1_D1 = pd.merge(D1, A1, on="ref_Index", how="inner")
    A1_D2 = pd.merge(D2, A1, on="ref_Index", how="inner")
    A1_D3 = pd.merge(D3, A1, on="ref_Index", how="inner")
    A1_D4 = pd.merge(D4, A1, on="ref_Index", how="inner")
    A1_D5 = pd.merge(D5, A1, on="ref_Index", how="inner")
    A1_D6 = pd.merge(D6, A1, on="ref_Index", how="inner")

    # define A2 as the 1st cell and repeat merging again
    A1 = A1.rename(columns={'Cell1': 'Cell2'})
    A2 = A2.rename(columns={'Cell2': 'Cell1'})

    A2_A3 = pd.merge(A2, A3, on="ref_Index", how="inner")
    A2_M1 = pd.merge(A2, M1, on="ref_Index", how="inner")
    A2_M2 = pd.merge(A2, M2, on="ref_Index", how="inner")
    A2_M3 = pd.merge(A2, M3, on="ref_Index", how="inner")
    A2_M4 = pd.merge(A2, M4, on="ref_Index", how="inner")
    A2_M5 = pd.merge(A2, M5, on="ref_Index", how="inner")
    A2_M6 = pd.merge(A2, M6, on="ref_Index", how="inner")
    A2_D1 = pd.merge(A2, D1, on="ref_Index", how="inner")
    A2_D2 = pd.merge(A2, D2, on="ref_Index", how="inner")
    A2_D3 = pd.merge(A2, D3, on="ref_Index", how="inner")
    A2_D4 = pd.merge(A2, D4, on="ref_Index", how="inner")
    A2_D5 = pd.merge(A2, D5, on="ref_Index", how="inner")
    A2_D6 = pd.merge(A2, D6, on="ref_Index", how="inner")

    # define A3 as the 1st cell and repeat merging again
    A2 = A2.rename(columns={'Cell1': 'Cell2'})
    A3 = A3.rename(columns={'Cell2': 'Cell1'})

    A3_M1 = pd.merge(M1, A3, on="ref_Index", how="inner")
    A3_M2 = pd.merge(M2, A3, on="ref_Index", how="inner")
    A3_M3 = pd.merge(M3, A3, on="ref_Index", how="inner")
    A3_M4 = pd.merge(M4, A3, on="ref_Index", how="inner")
    A3_M5 = pd.merge(M5, A3, on="ref_Index", how="inner")
    A3_M6 = pd.merge(M6, A3, on="ref_Index", how="inner")
    A3_D1 = pd.merge(D1, A3, on="ref_Index", how="inner")
    A3_D2 = pd.merge(D2, A3, on="ref_Index", how="inner")
    A3_D3 = pd.merge(D3, A3, on="ref_Index", how="inner")
    A3_D4 = pd.merge(D4, A3, on="ref_Index", how="inner")
    A3_D5 = pd.merge(D5, A3, on="ref_Index", how="inner")
    A3_D6 = pd.merge(D6, A3, on="ref_Index", how="inner")

    # list the neighbours  A&M
    Combination_NBR = pd.concat(
        [A1_A2, A1_A3, A2_A3, A1_M1, A1_M2, A1_M3, A1_M4, A1_M5, A1_M6, A2_M1, A2_M2, A2_M3, A2_M4, A2_M5, A2_M6,
         A3_M1, A3_M2, A3_M3, A3_M4, A3_M5, A3_M6], sort=False)

    # list the Missing     A&D
    Combination_Missing = pd.concat(
        [A1_D1, A1_D2, A1_D3, A1_D4, A1_D5, A1_D6, A2_D1, A2_D2, A2_D3, A2_D4, A2_D5, A2_D6, A3_D1,
         A3_D2, A3_D3, A3_D4, A3_D5, A3_D6], sort=False)

    Combination_NBR['State'] = 'Neighbour'
    Combination_Missing['State'] = 'Missed'

    # concat, bt7tohom wra ba3d coloum by coloum
    Combination = pd.concat([Combination_NBR, Combination_Missing], sort=False)
    Combination = Combination[['Cell1', 'Cell2', 'State']]

    #     print(Combination)

    Combination.to_csv('Combination.csv', index=True)
    Combination = pd.read_csv('Combination.csv')

    Combination = Combination.groupby(['Cell1', 'Cell2', 'State'])
    Combination = Combination.count().reset_index()
    Combination = Combination.rename(columns={'Unnamed: 0': 'Sum'})
    Combination.to_csv('Combination.csv', index=True)

    l = Combination.size
    Final = pd.DataFrame(columns=['Cell1', 'Cell2', 'Sum_NBR', 'SUM_Miss'])
    for i in range(Combination.shape[0]):  # loop on row
        Sum_NBR = 0
        Sum_Miss = 0
        for j in range(Combination.shape[0]):
            if (Combination['Cell1'][i] == Combination['Cell1'][j] and Combination['Cell2'][i] == Combination['Cell2'][
                j]) \
                    or (Combination['Cell1'][i] == Combination['Cell2'][j] and Combination['Cell2'][i] ==
                        Combination['Cell1'][j]):
                if Combination['State'][j] == 'Neighbour':
                    Sum_NBR += Combination['Sum'][j]
                else:
                    Sum_Miss += Combination['Sum'][j]
        Final = Final.append({'Cell1': Combination['Cell1'][i],
                              'Cell2': Combination['Cell2'][i],
                              'Sum_NBR': Sum_NBR, 'SUM_Miss': Sum_Miss}, ignore_index=True)

    # frozen set 34an amn3 edaft ay 7aga leha b3d kda
    Final = Final[~Final[['Cell1', 'Cell2']].apply(frozenset, axis=1).duplicated()]
    Final.to_csv('test.csv', index=True)
    Final = Final.reset_index()
    Final.to_csv('New.csv')

    Analysiss = pd.DataFrame(columns=['Cell1', 'Cell2', 'Sum_NBR', 'SUM_Miss', 'State'])

    for k in range(Final.shape[0]):
        if Final['SUM_Miss'][k] + Final['Sum_NBR'][k] < ceil(10):
            Analysiss = Analysiss.append({'Cell1': Final['Cell1'][k],
                                          'Cell2': Final['Cell2'][k],
                                          'Sum_NBR': Final['Sum_NBR'][k],
                                          'SUM_Miss': Final['SUM_Miss'][k],
                                          'State': "Can not decide"}, ignore_index=True)
        else:
            if Final['Sum_NBR'][k] == 0 and Final['SUM_Miss'][k] > 0:
                Analysiss = Analysiss.append({'Cell1': Final['Cell1'][k],
                                              'Cell2': Final['Cell2'][k],
                                              'Sum_NBR': Final['Sum_NBR'][k],
                                              'SUM_Miss': Final['SUM_Miss'][k],
                                              'State': "Missed Neighbour"}, ignore_index=True)

            elif Final['SUM_Miss'][k] / Final['Sum_NBR'][k] < 0.4:
                Analysiss = Analysiss.append({'Cell1': Final['Cell1'][k],
                                              'Cell2': Final['Cell2'][k],
                                              'Sum_NBR': Final['Sum_NBR'][k],
                                              'SUM_Miss': Final['SUM_Miss'][k],
                                              'State': "Normal Relation"}, ignore_index=True)
            else:
                Analysiss = Analysiss.append({'Cell1': Final['Cell1'][k],
                                              'Cell2': Final['Cell2'][k],
                                              'Sum_NBR': Final['Sum_NBR'][k],
                                              'SUM_Miss': Final['SUM_Miss'][k],
                                              'State': "Missed Neighbour"}, ignore_index=True)
    # print(Analysiss)
    Analysiss.to_csv('Missing_NBR_Analysis_test.csv')


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


def calc(x, Drive_Test_File):
    Cell_file = pd.read_csv(x)
    # Drive_Test_File = pd.read_csv(y)
    Cells = Cell_file[['Region', 'Cell', 'Lat', 'Lon', 'SC', 'UARFCN']]
    drive_test = Drive_Test_File[['Latitude', 'Longitude',
                                  'Categorized PSC:A1', 'Categorized UARFCN_DL:A1',
                                  'Categorized PSC:A2', 'Categorized UARFCN_DL:A2',
                                  'Categorized PSC:A3', 'Categorized UARFCN_DL:A3',
                                  'Categorized PSC:D1', 'Categorized UARFCN_DL:D1',
                                  'Categorized PSC:D2', 'Categorized UARFCN_DL:D2',
                                  'Categorized PSC:D3', 'Categorized UARFCN_DL:D3',
                                  'Categorized PSC:D4', 'Categorized UARFCN_DL:D4',
                                  'Categorized PSC:D5', 'Categorized UARFCN_DL:D5',
                                  'Categorized PSC:D6', 'Categorized UARFCN_DL:D6',
                                  'Categorized PSC:M1', 'Categorized UARFCN_DL:M1',
                                  'Categorized PSC:M2', 'Categorized UARFCN_DL:M2',
                                  'Categorized PSC:M3', 'Categorized UARFCN_DL:M3',
                                  'Categorized PSC:M4', 'Categorized UARFCN_DL:M4',
                                  'Categorized PSC:M5', 'Categorized UARFCN_DL:M5',
                                  'Categorized PSC:M6', 'Categorized UARFCN_DL:M6']]

    drive_test = drive_test.iloc[:, :].reset_index()  # get all data

    # Drop NAN columns#
    drive_test = drive_test[np.isfinite(drive_test['Latitude'])]
    drive_test = drive_test[np.isfinite(drive_test['Longitude'])]
    drive_test = drive_test.reset_index()

    # get data
    A1 = pd.DataFrame(columns=['ref_Index', 'Cell1', 'Distance', 'drive_lat', 'drive_lon', 'Lat_Cell', 'Lon_Cell'])
    A2 = pd.DataFrame(columns=['ref_Index', 'Cell2', 'Distance', 'drive_lat', 'drive_lon', 'Lat_Cell', 'Lon_Cell'])
    A3 = pd.DataFrame(columns=['ref_Index', 'Cell2', 'Distance', 'drive_lat', 'drive_lon', 'Lat_Cell', 'Lon_Cell'])
    D1 = pd.DataFrame(columns=['ref_Index', 'Cell2', 'Distance', 'drive_lat', 'drive_lon', 'Lat_Cell', 'Lon_Cell'])
    D2 = pd.DataFrame(columns=['ref_Index', 'Cell2', 'Distance', 'drive_lat', 'drive_lon', 'Lat_Cell', 'Lon_Cell'])
    D3 = pd.DataFrame(columns=['ref_Index', 'Cell2', 'Distance', 'drive_lat', 'drive_lon', 'Lat_Cell', 'Lon_Cell'])
    D4 = pd.DataFrame(columns=['ref_Index', 'Cell2', 'Distance', 'drive_lat', 'drive_lon', 'Lat_Cell', 'Lon_Cell'])
    D5 = pd.DataFrame(columns=['ref_Index', 'Cell2', 'Distance', 'drive_lat', 'drive_lon', 'Lat_Cell', 'Lon_Cell'])
    D6 = pd.DataFrame(columns=['ref_Index', 'Cell2', 'Distance', 'drive_lat', 'drive_lon', 'Lat_Cell', 'Lon_Cell'])
    M1 = pd.DataFrame(columns=['ref_Index', 'Cell2', 'Distance', 'drive_lat', 'drive_lon', 'Lat_Cell', 'Lon_Cell'])
    M2 = pd.DataFrame(columns=['ref_Index', 'Cell2', 'Distance', 'drive_lat', 'drive_lon', 'Lat_Cell', 'Lon_Cell'])
    M3 = pd.DataFrame(columns=['ref_Index', 'Cell2', 'Distance', 'drive_lat', 'drive_lon', 'Lat_Cell', 'Lon_Cell'])
    M4 = pd.DataFrame(columns=['ref_Index', 'Cell2', 'Distance', 'drive_lat', 'drive_lon', 'Lat_Cell', 'Lon_Cell'])
    M5 = pd.DataFrame(columns=['ref_Index', 'Cell2', 'Distance', 'drive_lat', 'drive_lon', 'Lat_Cell', 'Lon_Cell'])
    M6 = pd.DataFrame(columns=['ref_Index', 'Cell2', 'Distance', 'drive_lat', 'drive_lon', 'Lat_Cell', 'Lon_Cell'])
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

    #
    for i in range(0, drive_test.shape[0], 1):
        drive_sample = pd.DataFrame({'drive_lat': drive_test['Latitude'][i],
                                     'drive_lon': drive_test['Longitude'][i],
                                     'PSC_A1': drive_test['Categorized PSC:A1'][i],
                                     'UARFCN_A1': drive_test['Categorized UARFCN_DL:A1'][i],
                                     'PSC_A2': drive_test['Categorized PSC:A2'][i],
                                     'UARFCN_A2': drive_test['Categorized UARFCN_DL:A2'][i],
                                     'PSC_A3': drive_test['Categorized PSC:A3'][i],
                                     'UARFCN_A3': drive_test['Categorized UARFCN_DL:A3'][i],
                                     'PSC_D1': drive_test['Categorized PSC:D1'][i],
                                     'UARFCN_D1': drive_test['Categorized UARFCN_DL:D1'][i],
                                     'PSC_D2': drive_test['Categorized PSC:D2'][i],
                                     'UARFCN_D2': drive_test['Categorized UARFCN_DL:D2'][i],
                                     'PSC_D3': drive_test['Categorized PSC:D3'][i],
                                     'UARFCN_D3': drive_test['Categorized UARFCN_DL:D3'][i],
                                     'PSC_D4': drive_test['Categorized PSC:D4'][i],
                                     'UARFCN_D4': drive_test['Categorized UARFCN_DL:D4'][i],
                                     'PSC_D5': drive_test['Categorized PSC:D5'][i],
                                     'UARFCN_D5': drive_test['Categorized UARFCN_DL:D5'][i],
                                     'PSC_D6': drive_test['Categorized PSC:D6'][i],
                                     'UARFCN_D6': drive_test['Categorized UARFCN_DL:D6'][i],
                                     'PSC_M1': drive_test['Categorized PSC:M1'][i],
                                     'UARFCN_M1': drive_test['Categorized UARFCN_DL:M1'][i],
                                     'PSC_M2': drive_test['Categorized PSC:M2'][i],
                                     'UARFCN_M2': drive_test['Categorized UARFCN_DL:M2'][i],
                                     'PSC_M3': drive_test['Categorized PSC:M3'][i],
                                     'UARFCN_M3': drive_test['Categorized UARFCN_DL:M3'][i],
                                     'PSC_M4': drive_test['Categorized PSC:M4'][i],
                                     'UARFCN_M4': drive_test['Categorized UARFCN_DL:M4'][i],
                                     'PSC_M5': drive_test['Categorized PSC:M5'][i],
                                     'UARFCN_M5': drive_test['Categorized UARFCN_DL:M5'][i],
                                     'PSC_M6': drive_test['Categorized PSC:M6'][i],
                                     'UARFCN_M6': drive_test['Categorized UARFCN_DL:M6'][i]},
                                    columns=['drive_lat', 'drive_lon',
                                             'PSC_A1', 'UARFCN_A1', 'PSC_A2', 'UARFCN_A2', 'PSC_A3', 'UARFCN_A3',
                                             'PSC_D1', 'UARFCN_D1', 'PSC_D2', 'UARFCN_D2', 'PSC_D3', 'UARFCN_D3',
                                             'PSC_D4', 'UARFCN_D4', 'PSC_D5', 'UARFCN_D5', 'PSC_D6', 'UARFCN_D6',
                                             'PSC_M1', 'UARFCN_M1', 'PSC_M2', 'UARFCN_M2', 'PSC_M3', 'UARFCN_M3',
                                             'PSC_M4', 'UARFCN_M4', 'PSC_M5', 'UARFCN_M5', 'PSC_M6', 'UARFCN_M6'],
                                    index=[0])

        drive_sample = drive_sample.iloc[np.full(Cells_size, 0)]
        drive_sample = drive_sample.reset_index()
        Cells_A1 = Cells
        Cells_A2 = Cells
        Cells_A3 = Cells
        Cells_D1 = Cells
        Cells_D2 = Cells
        Cells_D3 = Cells
        Cells_D4 = Cells
        Cells_D5 = Cells
        Cells_D6 = Cells
        Cells_M1 = Cells
        Cells_M2 = Cells
        Cells_M3 = Cells
        Cells_M4 = Cells
        Cells_M5 = Cells
        Cells_M6 = Cells

        # get A1
        Cells_A1['Cond'] = np.where((abs(drive_sample['drive_lat'] - Cells['Lat']) < 0.1) & (
                    abs(drive_sample['drive_lon'] - Cells['Lon']) < 0.1) & (drive_sample['PSC_A1'] == Cells['SC']) & (
                                                drive_sample['UARFCN_A1'] == Cells['UARFCN']), 'True', 'False')
        Cells_A1 = Cells_A1[Cells_A1['Cond'] == "True"].reset_index()
        if Cells_A1.shape[0] > 0:
            A1 = A1.append({'ref_Index': int(i), 'Cell1': Cells_A1['Cell'][0],
                            'Lat_Cell': Cells_A1['Lat'][0],
                            'Lon_Cell': Cells_A1['Lon'][0],
                            'drive_lat': drive_sample['drive_lat'][0],
                            'drive_lon': drive_sample['drive_lon'][0]},
                           ignore_index=True)
            # get A2
        Cells_A2['Cond'] = np.where((abs(drive_sample['drive_lat'] - Cells['Lat']) < 0.1) & (
                    abs(drive_sample['drive_lon'] - Cells['Lon']) < 0.1) & (drive_sample['PSC_A2'] == Cells['SC']) & (
                                                drive_sample['UARFCN_A2'] == Cells['UARFCN']), 'True', 'False')
        Cells_A2 = Cells_A2[Cells_A2['Cond'] == "True"].reset_index()
        if Cells_A2.shape[0] > 0:
            A2 = A2.append({'ref_Index': int(i), 'Cell2': Cells_A2['Cell'][0],
                            'drive_lat': drive_sample['drive_lat'][0],
                            'drive_lon': drive_sample['drive_lon'][0],
                            'Lat_Cell': Cells_A2['Lat'][0],
                            'Lon_Cell': Cells_A2['Lon'][0]},
                           ignore_index=True)

        # get A3
        Cells_A3['Cond'] = np.where((abs(drive_sample['drive_lat'] - Cells['Lat']) < 0.1) & (
                    abs(drive_sample['drive_lon'] - Cells['Lon']) < 0.1) & (drive_sample['PSC_A3'] == Cells['SC']) & (
                                                drive_sample['UARFCN_A3'] == Cells['UARFCN']), 'True', 'False')
        Cells_A3 = Cells_A3[Cells_A3['Cond'] == "True"].reset_index()
        if Cells_A3.shape[0] > 0:
            A3 = A3.append({'ref_Index': int(i), 'Cell2': Cells_A3['Cell'][0],
                            'drive_lat': drive_sample['drive_lat'][0],
                            'drive_lon': drive_sample['drive_lon'][0],
                            'Lat_Cell': Cells_A3['Lat'][0],
                            'Lon_Cell': Cells_A3['Lon'][0]},
                           ignore_index=True)

        # get D1
        Cells_D1['Cond'] = np.where((abs(drive_sample['drive_lat'] - Cells['Lat']) < 0.1) & (
                    abs(drive_sample['drive_lon'] - Cells['Lon']) < 0.1) & (drive_sample['PSC_D1'] == Cells['SC']) & (
                                                drive_sample['UARFCN_D1'] == Cells['UARFCN']), 'True', 'False')
        Cells_D1 = Cells_D1[Cells_D1['Cond'] == "True"].reset_index()
        if Cells_D1.shape[0] > 0:
            D1 = D1.append({'ref_Index': int(i), 'Cell2': Cells_D1['Cell'][0],
                            'drive_lat': drive_sample['drive_lat'][0],
                            'drive_lon': drive_sample['drive_lon'][0],
                            'Lat_Cell': Cells_D1['Lat'][0],
                            'Lon_Cell': Cells_D1['Lon'][0]},
                           ignore_index=True)

            # get D2
        Cells_D2['Cond'] = np.where((abs(drive_sample['drive_lat'] - Cells['Lat']) < 0.1) & (
                    abs(drive_sample['drive_lon'] - Cells['Lon']) < 0.1) & (drive_sample['PSC_D2'] == Cells['SC']) & (
                                                drive_sample['UARFCN_D2'] == Cells['UARFCN']), 'True', 'False')
        Cells_D2 = Cells_D2[Cells_D2['Cond'] == "True"].reset_index()
        if Cells_D2.shape[0] > 0:
            D2 = D2.append({'ref_Index': int(i), 'Cell2': Cells_D2['Cell'][0],
                            'drive_lat': drive_sample['drive_lat'][0],
                            'drive_lon': drive_sample['drive_lon'][0],
                            'Lat_Cell': Cells_D2['Lat'][0],
                            'Lon_Cell': Cells_D2['Lon'][0]},
                           ignore_index=True)

        # get D3
        Cells_D3['Cond'] = np.where((abs(drive_sample['drive_lat'] - Cells['Lat']) < 0.1) & (
                    abs(drive_sample['drive_lon'] - Cells['Lon']) < 0.1) & (drive_sample['PSC_D3'] == Cells['SC']) & (
                                                drive_sample['UARFCN_D3'] == Cells['UARFCN']), 'True', 'False')
        Cells_D3 = Cells_D3[Cells_D3['Cond'] == "True"].reset_index()
        if Cells_D3.shape[0] > 0:
            D3 = D3.append({'ref_Index': int(i), 'Cell2': Cells_D3['Cell'][0],
                            'drive_lat': drive_sample['drive_lat'][0],
                            'drive_lon': drive_sample['drive_lon'][0],
                            'Lat_Cell': Cells_D3['Lat'][0],
                            'Lon_Cell': Cells_D3['Lon'][0]},
                           ignore_index=True)
        # get D4
        Cells_D4['Cond'] = np.where((abs(drive_sample['drive_lat'] - Cells['Lat']) < 0.1) & (
                    abs(drive_sample['drive_lon'] - Cells['Lon']) < 0.1) & (drive_sample['PSC_D4'] == Cells['SC']) & (
                                                drive_sample['UARFCN_D4'] == Cells['UARFCN']), 'True', 'False')
        Cells_D4 = Cells_D4[Cells_D4['Cond'] == "True"].reset_index()
        if Cells_D4.shape[0] > 0:
            D4 = D4.append({'ref_Index': int(i), 'Cell2': Cells_D4['Cell'][0],
                            'drive_lat': drive_sample['drive_lat'][0],
                            'drive_lon': drive_sample['drive_lon'][0],
                            'Lat_Cell': Cells_D4['Lat'][0],
                            'Lon_Cell': Cells_D4['Lon'][0]},
                           ignore_index=True)

            # get D5
        Cells_D5['Cond'] = np.where((abs(drive_sample['drive_lat'] - Cells['Lat']) < 0.1) & (
                    abs(drive_sample['drive_lon'] - Cells['Lon']) < 0.1) & (drive_sample['PSC_D5'] == Cells['SC']) & (
                                                drive_sample['UARFCN_D5'] == Cells['UARFCN']), 'True', 'False')
        Cells_D5 = Cells_D5[Cells_D5['Cond'] == "True"].reset_index()
        if Cells_D5.shape[0] > 0:
            D5 = D5.append({'ref_Index': int(i), 'Cell2': Cells_D5['Cell'][0],
                            'drive_lat': drive_sample['drive_lat'][0],
                            'drive_lon': drive_sample['drive_lon'][0],
                            'Lat_Cell': Cells_D5['Lat'][0],
                            'Lon_Cell': Cells_D5['Lon'][0]},
                           ignore_index=True)

        # get D6
        Cells_D6['Cond'] = np.where((abs(drive_sample['drive_lat'] - Cells['Lat']) < 0.1) & (
                    abs(drive_sample['drive_lon'] - Cells['Lon']) < 0.1) & (drive_sample['PSC_D6'] == Cells['SC']) & (
                                                drive_sample['UARFCN_D6'] == Cells['UARFCN']), 'True', 'False')
        Cells_D6 = Cells_D6[Cells_D6['Cond'] == "True"].reset_index()
        if Cells_D6.shape[0] > 0:
            D6 = D6.append({'ref_Index': int(i), 'Cell2': Cells_D6['Cell'][0],
                            'drive_lat': drive_sample['drive_lat'][0],
                            'drive_lon': drive_sample['drive_lon'][0],
                            'Lat_Cell': Cells_D6['Lat'][0],
                            'Lon_Cell': Cells_D6['Lon'][0]},
                           ignore_index=True)

        # get M1
        Cells_M1['Cond'] = np.where((abs(drive_sample['drive_lat'] - Cells['Lat']) < 0.1) & (
                    abs(drive_sample['drive_lon'] - Cells['Lon']) < 0.1) & (drive_sample['PSC_M1'] == Cells['SC']) & (
                                                drive_sample['UARFCN_M1'] == Cells['UARFCN']), 'True', 'False')
        Cells_M1 = Cells_M1[Cells_M1['Cond'] == "True"].reset_index()
        if Cells_M1.shape[0] > 0:
            M1 = M1.append({'ref_Index': int(i), 'Cell2': Cells_M1['Cell'][0],
                            'drive_lat': drive_sample['drive_lat'][0],
                            'drive_lon': drive_sample['drive_lon'][0],
                            'Lat_Cell': Cells_M1['Lat'][0],
                            'Lon_Cell': Cells_M1['Lon'][0]},
                           ignore_index=True)

            # get M2
        Cells_M2['Cond'] = np.where((abs(drive_sample['drive_lat'] - Cells['Lat']) < 0.1) & (
                    abs(drive_sample['drive_lon'] - Cells['Lon']) < 0.1) & (drive_sample['PSC_M2'] == Cells['SC']) & (
                                                drive_sample['UARFCN_M2'] == Cells['UARFCN']), 'True', 'False')
        Cells_M2 = Cells_M2[Cells_M2['Cond'] == "True"].reset_index()
        if Cells_M2.shape[0] > 0:
            M2 = M2.append({'ref_Index': int(i), 'Cell2': Cells_M2['Cell'][0],
                            'drive_lat': drive_sample['drive_lat'][0],
                            'drive_lon': drive_sample['drive_lon'][0],
                            'Lat_Cell': Cells_M2['Lat'][0],
                            'Lon_Cell': Cells_M2['Lon'][0]},
                           ignore_index=True)

        # get M3
        Cells_M3['Cond'] = np.where((abs(drive_sample['drive_lat'] - Cells['Lat']) < 0.1) & (
                    abs(drive_sample['drive_lon'] - Cells['Lon']) < 0.1) & (drive_sample['PSC_M3'] == Cells['SC']) & (
                                                drive_sample['UARFCN_M3'] == Cells['UARFCN']), 'True', 'False')
        Cells_M3 = Cells_M3[Cells_M3['Cond'] == "True"].reset_index()
        if Cells_M3.shape[0] > 0:
            M3 = M3.append({'ref_Index': int(i), 'Cell2': Cells_M3['Cell'][0],
                            'drive_lat': drive_sample['drive_lat'][0],
                            'drive_lon': drive_sample['drive_lon'][0],
                            'Lat_Cell': Cells_M3['Lat'][0],
                            'Lon_Cell': Cells_M3['Lon'][0]},
                           ignore_index=True)

        # get M4
        Cells_M4['Cond'] = np.where((abs(drive_sample['drive_lat'] - Cells['Lat']) < 0.1) & (
                    abs(drive_sample['drive_lon'] - Cells['Lon']) < 0.1) & (drive_sample['PSC_M4'] == Cells['SC']) & (
                                                drive_sample['UARFCN_M4'] == Cells['UARFCN']), 'True', 'False')
        Cells_M4 = Cells_M4[Cells_M4['Cond'] == "True"].reset_index()
        if Cells_M4.shape[0] > 0:
            M4 = M4.append({'ref_Index': int(i), 'Cell2': Cells_M4['Cell'][0],
                            'drive_lat': drive_sample['drive_lat'][0],
                            'drive_lon': drive_sample['drive_lon'][0],
                            'Lat_Cell': Cells_M4['Lat'][0],
                            'Lon_Cell': Cells_M4['Lon'][0]},
                           ignore_index=True)

        # get M5
        Cells_M5['Cond'] = np.where((abs(drive_sample['drive_lat'] - Cells['Lat']) < 0.1) & (
                    abs(drive_sample['drive_lon'] - Cells['Lon']) < 0.1) & (drive_sample['PSC_M5'] == Cells['SC']) & (
                                                drive_sample['UARFCN_M5'] == Cells['UARFCN']), 'True', 'False')
        Cells_M5 = Cells_M5[Cells_M5['Cond'] == "True"].reset_index()
        if Cells_M5.shape[0] > 0:
            M5 = M5.append({'ref_Index': int(i), 'Cell2': Cells_M5['Cell'][0],
                            'drive_lat': drive_sample['drive_lat'][0],
                            'drive_lon': drive_sample['drive_lon'][0],
                            'Lat_Cell': Cells_M5['Lat'][0],
                            'Lon_Cell': Cells_M5['Lon'][0]},
                           ignore_index=True)
        # get M6
        Cells_M6['Cond'] = np.where((abs(drive_sample['drive_lat'] - Cells['Lat']) < 0.1) & (
                    abs(drive_sample['drive_lon'] - Cells['Lon']) < 0.1) & (drive_sample['PSC_M6'] == Cells['SC']) & (
                                                drive_sample['UARFCN_M6'] == Cells['UARFCN']), 'True', 'False')
        Cells_M6 = Cells_M6[Cells_M6['Cond'] == "True"].reset_index()
        if Cells_M6.shape[0] > 0:
            M6 = M6.append({'ref_Index': int(i), 'Cell2': Cells_M6['Cell'][0],
                            'drive_lat': drive_sample['drive_lat'][0],
                            'drive_lon': drive_sample['drive_lon'][0],
                            'Lat_Cell': Cells_M6['Lat'][0],
                            'Lon_Cell': Cells_M6['Lon'][0]},
                           ignore_index=True)

    return A1, A2, A3, D1, D2, D3, D4, D5, D6, M1, M2, M3, M4, M5, M6

