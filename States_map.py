
# IMPORTS
import gmaps
import os
import numpy as np
import pandas as pd
from ipywidgets.embed import embed_minimal_html


def generate_heat_map(Final_Report, total_box, info_box):
    # CONFIGURATION
    API_KEY = 'AIzaSyBw-v4Mq3Jg-RDYv3N_9OVaqnOcXY-4WRY'
    gmaps.configure(api_key=API_KEY)

    data_problems = pd.read_csv(Final_Report)
    file_box = pd.read_csv(total_box)
    MAP_TYPE = 'SATELLITE'  # or use any available view ['HYBRID', 'ROADMAP', 'TERRAIN', 'SATELLITE']

    X = 'Lat_drive'
    Y = 'Lon_drive'
    COL = 'States'

    FIG_LAYOUT = {
        'width': '1100px',
        'height': '800px',
        'border': '1px solid black',
        'padding': '1px'
    }

    Cell_ID = file_box['Cell_ID']
    X0 = file_box['Lat_Cell']
    Y0 = file_box['Lon_Cell']
    theta = file_box['Ant_Dirction']
    Sum_Nearest = file_box['Nearest_per']
    Sum_overshooting = file_box['overshooting_per']
    Sum_Blocked = file_box['Blocked_per']
    Sum_Bad_Coverge = file_box['Bad_Coverge_per']
    Sum_Loaded = file_box['Loaded_per']
    Sum_cross_sector = file_box['cross_sector_per']
    Sum_back_lobe = file_box['back_lobe_per']

    Cell_ID = Cell_ID.values.tolist()
    X0 = X0.values.tolist()
    Y0 = Y0.values.tolist()
    theta = theta.values.tolist()
    Sum_Nearest = Sum_Nearest.values.tolist()
    Sum_overshooting = Sum_overshooting.values.tolist()
    Sum_Blocked = Sum_Blocked.values.tolist()
    Sum_cross_sector = Sum_cross_sector.values.tolist()
    Sum_back_lobe = Sum_back_lobe.values.tolist()

    fig2 = gmaps.figure(layout=FIG_LAYOUT, map_type=MAP_TYPE)

    def triangle_coord(Cell_ID, X0, Y0, theta, Sum_Nearest, Sum_overshooting, Sum_Blocked,
                       Sum_Bad_Coverge, Sum_Loaded, Sum_cross_sector, Sum_back_lobe, color):
        theta1 = np.add(theta, 60)
        theta2 = np.add(theta, 120)
        d = 0.0002
        angle1 = np.radians(theta1)
        angle2 = np.radians(theta2)
        X1 = X0 + d * np.sin(angle1)
        Y1 = Y0 + d * np.cos(angle1)
        X2 = X0 + d * np.sin(angle2)
        Y2 = Y0 + d * np.cos(angle2)
        for i in range(377):
            drawing = gmaps.drawing_layer(features=[gmaps.Polygon([(X0[i], Y0[i]), (X1[i], Y1[i]), (X2[i], Y2[i])],
                                                                  stroke_color='black', fill_color=color)],
                                          show_controls=False)
            nuclear_power_plants = [{'name': Cell_ID[i], 'location': (X1[i], Y1[i]),
                                     'Sum_Nearest': Sum_Nearest[i],
                                     'Sum_overshooting': Sum_overshooting[i],
                                     'Sum_Blocked': Sum_Blocked[i],
                                     'Sum_Bad_Coverge': Sum_Bad_Coverge[i],
                                     'Sum_Loaded': Sum_Loaded[i],
                                     'Sum_cross_sector': Sum_cross_sector[i],
                                     'Sum_back_lobe': Sum_back_lobe[i]}]

            plant_locations = [plant['location'] for plant in nuclear_power_plants]
            info_box_template = """
            <dl>
              <dt >Cell_id</dt><dd>{name}</dd>
              <dt>Nearest <span style="color:#00FFFF;" style='font-size:16px;'>&#9899;</span></dt><dd>{Sum_Nearest} %</dd>
              <dt>overshooting  <span style="color:#FF7D33;" style='font-size:16px;'>&#9899;</span></dt><dd>{Sum_overshooting} %</dd>
              <dt>Blocked  <span style="color:#FF00FF;" style='font-size:16px;'>&#9899;</span></dt><dd>{Sum_Blocked} %</dd>
              <dt>Bad_Coverge  <span style="color:#34FF33;" style='font-size:16px;'>&#9899;</span></dt><dd>{Sum_Bad_Coverge} %</dd>
              <dt>Loaded  <span style="color:#0000FF;" style='font-size:16px;'>&#9899;</span></dt><dd>{Sum_Loaded} %</dd>
              <dt>Cross_Sector  <span style="color:#FFFF00;" style='font-size:16px;'>&#9899;</span></dt><dd>{Sum_cross_sector} %</dd>
              <dt>Back_Lobe  <span style="color:#FF0000;" style='font-size:16px;'>&#9899;</span></dt><dd>{Sum_back_lobe} %</dd>          
            </dl>        
            """
            plant_info = [info_box_template.format(**plant) for plant in nuclear_power_plants]
            marker_layer = gmaps.marker_layer(plant_locations, info_box_content=plant_info)

            fig2.add_layer(marker_layer)
            fig2.add_layer(drawing)

    color = 'yellow'

    triangle_coord(Cell_ID, X0, Y0, theta, Sum_Nearest, Sum_overshooting, Sum_Blocked,
                   Sum_Bad_Coverge, Sum_Loaded, Sum_cross_sector, Sum_back_lobe, color)

    #########LOAD DATA

    DATA_SIZE = 26045

    # SAMPLES_CLASSIFICATIONS

    nearest_locations = []
    blocked_locations = []
    loaded_locations = []
    overshoot_locations = []
    Bad_Coverge_locations = []
    cross_sector_locations = []
    back_lobe_locations = []

    for i in range(DATA_SIZE):
        if (data_problems[COL][i] == 'Blocked'):
            blocked_locations.append((data_problems[X][i], data_problems[Y][i]))
        elif (data_problems[COL][i] == 'Nearest'):
            nearest_locations.append((data_problems[X][i], data_problems[Y][i]))
        elif (data_problems[COL][i] == 'Loaded'):
            loaded_locations.append((data_problems[X][i], data_problems[Y][i]))
        elif (data_problems[COL][i] == 'serving cell is overshooting'):
            overshoot_locations.append((data_problems[X][i], data_problems[Y][i]))
        elif (data_problems[COL][i] == 'Bad Coverge'):
            Bad_Coverge_locations.append((data_problems[X][i], data_problems[Y][i]))
        elif (data_problems[COL][i] == 'Diff_sector'):
            cross_sector_locations.append((data_problems[X][i], data_problems[Y][i]))
        elif (data_problems[COL][i] == 'serving cell is back lobe'):
            back_lobe_locations.append((data_problems[X][i], data_problems[Y][i]))

            # FIGURE

    nearest_layer = gmaps.symbol_layer(nearest_locations, fill_color='black', stroke_color=(0, 255, 255), scale=3)
    overshoot_layer = gmaps.symbol_layer(overshoot_locations, fill_color='black', stroke_color=(255, 125, 51), scale=3)
    blocked_layer = gmaps.symbol_layer(blocked_locations, fill_color='black', stroke_color=(255, 0, 255), scale=3)
    Bad_Coverge_layer = gmaps.symbol_layer(Bad_Coverge_locations, fill_color='black', stroke_color=(52, 255, 51),
                                           scale=3)
    loaded_layer = gmaps.symbol_layer(loaded_locations, fill_color='black', stroke_color=(0, 0, 255), scale=3)
    cross_sector_layer = gmaps.symbol_layer(cross_sector_locations, fill_color='black', stroke_color=(255, 255, 0),
                                            scale=3)
    back_lobe_layer = gmaps.symbol_layer(back_lobe_locations, fill_color='black', stroke_color=(255, 0, 0), scale=3)

    fig2.add_layer(Bad_Coverge_layer)
    fig2.add_layer(blocked_layer)
    fig2.add_layer(overshoot_layer)
    fig2.add_layer(nearest_layer)
    fig2.add_layer(loaded_layer)
    fig2.add_layer(cross_sector_layer)
    fig2.add_layer(back_lobe_layer)
    fig2

    embed_minimal_html('export.html', views=[fig2])
