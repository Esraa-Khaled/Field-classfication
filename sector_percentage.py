
## get percentage of each sector

import pandas as pd
def create_sector_perecentage(Final_Report, serving_cell_is_back_lobe, Diff_Sector):
    data = pd.read_csv('Final_Report.csv')
    back = pd.read_csv('serving_cell_is_back_lobe.csv')
    cross = pd.read_csv('Diff_Sector.csv')
    Final = pd.DataFrame(columns=['Cell_ID', 'Sum_Nearest', 'Sum_overshooting',
                                  'Sum_Blocked', 'Sum_Bad_Coverge', 'Sum_Loaded',
                                  'Sum_cross_sector', 'Sum_back_lobe'])

    ALL_states = pd.concat([data, cross, back], sort=False)

    n = ALL_states.groupby('States').size()
    z = ALL_states.groupby('Cell').size()
    z = z.reset_index()

    for i in range(z.shape[0]):
        Sum_Nearest = 0
        Sum_overshooting = 0
        Sum_Blocked = 0
        Sum_Bad_Coverge = 0
        Sum_Loaded = 0
        Sum_cross_sector = 0
        Sum_back_lobe = 0
        for j in range(data.shape[0]):
            if (z['Cell'][i] == data['Cell'][j]):
                if data['States'][j] == 'Nearest':
                    Sum_Nearest += 1
                elif data['States'][j] == 'serving cell is overshooting':
                    Sum_overshooting += 1
                elif data['States'][j] == 'Blocked':
                    Sum_Blocked += 1
                elif data['States'][j] == 'Bad Coverge':
                    Sum_Bad_Coverge += 1
                elif data['States'][j] == 'Loaded':
                    Sum_Loaded += 1
                elif data['States'][j] == 'Diff_sector':
                    Sum_cross_sector += 1
                elif data['States'][j] == 'serving cell is back lobe':
                    Sum_back_lobe += 1
        Final = Final.append({'Cell_ID': z['Cell'][i],
                              'Sum_Nearest': Sum_Nearest, 'Sum_overshooting': Sum_overshooting,
                              'Sum_Blocked': Sum_Blocked, 'Sum_Bad_Coverge': Sum_Bad_Coverge,
                              'Sum_Loaded': Sum_Loaded, 'Sum_cross_sector': Sum_cross_sector,
                              'Sum_back_lobe': Sum_back_lobe}, ignore_index=True)

    ###
    # Final = Final.apply(frozenset)
    Final = Final.reset_index()
    Final.to_csv('info_box.csv', index=True)
    ff = data.drop_duplicates(subset=['Cell', 'Ant_Dirction'])
    A1_A2 = pd.merge(Final, ff, left_on='Cell_ID', right_on='Cell', how="outer")
    box = A1_A2[['Cell_ID', 'Lat_Cell', 'Lon_Cell', 'Ant_Dirction', 'Sum_Nearest',
                 'Sum_overshooting', 'Sum_Blocked', 'Sum_Bad_Coverge', 'Sum_Loaded',
                 'Sum_cross_sector', 'Sum_back_lobe']]
    box = box.iloc[:, :].reset_index()
    box['SUM_col'] = box['Sum_Nearest'] + box['Sum_overshooting'] + box['Sum_Blocked'] + box['Sum_Bad_Coverge'] + box[
        'Sum_Loaded'] + box['Sum_cross_sector'] + box['Sum_back_lobe']

    box['Nearest_per'] = (box['Sum_Nearest'] / box['SUM_col']) * 100
    box['overshooting_per'] = (box['Sum_overshooting'] / box['SUM_col']) * 100
    box['Blocked_per'] = (box['Sum_Blocked'] / box['SUM_col']) * 100
    box['Bad_Coverge_per'] = (box['Sum_Bad_Coverge'] / box['SUM_col']) * 100
    box['Loaded_per'] = (box['Sum_Loaded'] / box['SUM_col']) * 100
    box['cross_sector_per'] = (box['Sum_cross_sector'] / box['SUM_col']) * 100
    box['back_lobe_per'] = (box['Sum_back_lobe'] / box['SUM_col']) * 100

    decimals = 2
    box['Nearest_per'] = box['Nearest_per'].apply(lambda x: round(x, decimals))
    box['overshooting_per'] = box['overshooting_per'].apply(lambda x: round(x, decimals))
    box['Blocked_per'] = box['Blocked_per'].apply(lambda x: round(x, decimals))
    box['Bad_Coverge_per'] = box['Bad_Coverge_per'].apply(lambda x: round(x, decimals))
    box['Loaded_per'] = box['Loaded_per'].apply(lambda x: round(x, decimals))
    box['cross_sector_per'] = box['cross_sector_per'].apply(lambda x: round(x, decimals))
    box['back_lobe_per'] = box['back_lobe_per'].apply(lambda x: round(x, decimals))

    box.to_csv('total_box.csv', index=True)








