from folium import folium

import pyspark
from pyspark import SparkContext
from pyspark import SparkFiles
from pyspark.sql import SparkSession
import pandas as pd
#permettant la manipulation et l'analyse des données

import numpy as np
#destinée à manipuler des matrices ou tableaux multidimensionnels ainsi que des fonctions mathématiques opérant sur ces tableaux

import matplotlib.pyplot as plt
#représenter des graphiques en 2D,regrouper les  fonctions qui servent à créer des graphiques et les personnaliser (travailler sur les axes, le type de graphique, sa forme et même rajouter du texte

spk = SparkSession.builder.appName('Paris Wifi Analysis').master("local[*]").getOrCreate()

spk.sparkContext.setLogLevel("ERROR")

spk_df = spk.read.options(inferSchema='True', header='True', delimiter=';').csv("paris-wifi.csv")
spk_df.printSchema()

# Drop the columns in the following list
columns_to_drop = ['code_site', 'temps_de_sessions_en_minutes', 'incomingzonelabel', 'incomingnetworklabel',
                   'device_operating_system_name_version', 'device_browser_name_version', 'donnee_entrante_go',
                   'donnee_sortante_gigaoctet', 'packetsin', 'packetsout', 'geo_shape']
spk_df_clean = spk_df.drop(*columns_to_drop)

print("Data Frame after cleaning")
spk_df_clean.printSchema()

spk_df_clean.groupby("cp", "device_portal_format").count().show()


spk_pd_df_clean = spk_df_clean.toPandas()

spk_pd_df_clean.info()
ax1 = spk_pd_df_clean['duration'].plot(kind='hist', bins=25, facecolor='lightblue')
ax1.set_title('duration__bytesin ')
ax1.set_xlabel('bytesin')
ax1.set_ylabel('duration')
plt.suptitle('')
plt.show()

DEPARTEMENT = ['AIN','AISNE','ALLIER','ALPES-DE-HAUTE-PROVENCE','HAUTES-ALPES','ALPES-MARITIMES','ARDECHE',
                   'ARDENNES','ARIEGE','AUBE','AUDE','AVEYRON','BOUCHES-DU-RHONE','CALVADOS','CANTAL','CHARENTE',
                   'CHARENTE-MARITIME','CHER','CORREZE','CORSE-DU-SUD','HAUTE-CORSE','COTE-D OR','COTES-D ARMOR',
                   'CREUSE','DORDOGNE','DOUBS','DROME','EURE','EURE-ET-LOIR','FINISTERE','GARD','HAUTE-GARONNE',
                   'GERS','GIRONDE','HERAULT','ILLE-ET-VILAINE','INDRE','INDRE-ET-LOIRE','ISERE','JURA','LANDES',
                   'LOIR-ET-CHER','LOIRE','HAUTE-LOIRE','LOIRE-ATLANTIQUE','LOIRET','LOT','LOT-ET-GARONNE','LOZERE',
                   'MAINE-ET-LOIRE','MANCHE','MARNE','HAUTE-MARNE','MAYENNE','MEURTHE-ET-MOSELLE','MEUSE','MORBIHAN',
                   'MOSELLE','NIEVRE','NORD','OISE','ORNE','PAS-DE-CALAIS','PUY-DE-DOME','PYRENEES-ATLANTIQUES',
                   'HAUTES-PYRENEES','PYRENEES-ORIENTALES','BAS-RHIN','HAUT-RHIN','RHONE','HAUTE-SAONE','SAONE-ET-LOIRE',
                   'SARTHE','SAVOIE','HAUTE-SAVOIE','PARIS','SEINE-MARITIME','SEINE-ET-MARNE','YVELINES','DEUX-SEVRES',
                   'SOMME','TARN','TARN-ET-GARONNE','VAR','VAUCLUSE','VENDEE','VIENNE','HAUTE-VIENNE','VOSGES','YONNE',
                   'TERRITOIRE DE BELFORT','ESSONNE','HAUTS-DE-SEINE','SEINE-SAINT-DENIS','VAL-DE-MARNE','VAL-D-OISE']

LATS = [46.0558,49.3334,46.2337,44.0622,44.3949,43.5615,44.4506,49.3656,42.5515,48.1816,43.0612,44.1649,43.3236,
                    49.0559,45.0304,45.4305,45.4651,47.0353,45.2125,41.5149,42.2339,47.2529,48.2628,46.0525,45.0615,47.0955,
                    44.4103,49.0649,48.2315,48.1540,43.5936,43.2131,43.4134,44.4931,43.3447,48.0916,46.4640,47.1529,45.1548,
                    46.4342,43.5756,47.3700,45.4337,45.0741,47.2141,47.5443,44.3727,44.2203,44.3102,47.2327,49.0446,48.5657,
                    48.0634,48.0848,48.4713,48.5922,47.5047,49.0214,47.0655,50.2650,49.2437,48.3725,50.2937,45.4333,43.1524,
                    43.0311,42.3600,48.4015,47.5131,45.5213,47.3828,46.3841,47.5940,45.2839,46.0204,48.5124,49.3918,48.3736,
                    48.4854,46.3320,49.5729,43.4707,44.0509,43.2738,43.5938,46.4029,46.3350,45.5330,48.1148,47.5023,47.3754,
                    48.3120,48.5050,48.5503,48.4639,49.0458]

LONGS = [5.2056,3.3330,3.1118,6.1438,6.1547,7.0659,4.2529,4.3827, 1.3014,4.0942,2.2451,2.4047,5.0511,
                    0.2149,2.4007,0.1206,0.4028,2.2928,1.5237,8.5917,9.1223,4.4620,2.5151,2.0108,0.4429,
                    6.2142,5.1005,0.5946,1.2213,4.0332,4.1049,1.1022,0.2712,0.3431,3.2202,1.3819,1.3433,0.4129,
                    5.3434,5.4152,0.4702,1.2546,4.0957,3.4823,1.4056,2.2039,1.3617,0.2737,3.3001,0.3351,1.1939,
                    4.1419,5.1335,0.3929,6.0954,5.2254,2.4836,6.3948,3.3017,3.1314,2.2531,0.0744,2.1719,3.0827,
                    0.4541,0.0950,2.3120,7.3305,7.1627,4.3829,6.0510,4.3232,0.1320,6.2637,6.2541,2.2032,1.0135,
                    2.5600,1.5030,0.1902,2.1640,2.0958,1.1655,6.1305,5.1110,1.1752,0.2737,1.1407,6.2250,3.3352,
                    6.5543,2.1435,2.1445,2.2841,2.2808,2.0752]

CAS = spk_df['duration']

import branca #Cette bibliothèque est une dérivée de folium,
               #qui hébergerait les fonctionnalités non spécifiques à la carte.

#coordonnées gps (lattitude et longuitude)
coords = (46.227638,2.213749)
map = folium.Map(location=coords, tiles='OpenStreetMap', zoom_start=6)


map.save(outfile='map.html')
map