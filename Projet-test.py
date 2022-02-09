import folium
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import isnan, when, count, col, lit, trim, avg, ceil
from pyspark.sql.types import StringType
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# creation d'une spark session
sc = SparkSession.builder.master("local[*]").getOrCreate()

feature = sc.read.csv("features.csv", inferSchema=True, header=True)
label = sc.read.csv("labels.csv", inferSchema=True, header=True)

print(feature.count())
print(label.count())
feature.printSchema()
label.printSchema()

# Fusionner les deux dataset
data = feature.join(label, on="id")
print(data.count())
data.printSchema()

# Affichage des 10 premieres lignes du dataframe
data.show(n=10)

# La fonction withColumn prend deux arguments -le nom de la nouvelle colonne et l'expression de la nouvelle colonne

data = data.withColumn('region_code', col('region_code').cast(StringType())).withColumn('district_code',
                                                                                        col('district_code').cast(
                                                                                            StringType()))
data.printSchema()

# supprimer les doublons
data = data.dropDuplicates(["id"])
data.count()

# en utilisant la fonction trim(), on peut supprimer les espaces blancs dans chaque de ces colonnes.
str_cols = [item[0] for item in data.dtypes if item[1].startswith('string')]

for cols in str_cols:
    data = data.withColumn(cols, trim(data[cols]))

# supprimer les colonnes avec les valeurs nulles a partir d'un certain seuil

data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data.columns if
             c not in {'date_recorded', 'public_meeting', 'permit'}]).show()

agg_row = data.select([(count(when(isnan(c) | col(c).isNull(), c)) / data.count()).alias(c) for c in data.columns if
                       c not in {'date_recorded', 'public_meeting', 'permit'}]).collect()
# convertir le résultat dans un dictionnaire pour une itération facile

agg_dict_list = [row.asDict() for row in agg_row]
agg_dict = agg_dict_list[0]

# on itére sur ce dictionnaire et stocke les noms de colonnes, qui ont nombre moyen de valeurs nulles plus qu'un
# seuil donné on a choisit 0.4 par rapport aux 40%

col_null = list({i for i in agg_dict if agg_dict[i] > 0.4})

print(agg_dict)

print(col_null)

data = data.drop(*col_null)

# Groupping et aggregation
# count utilise l'ordre asc
data.groupBy('recorded_by').count().show()
# comme on peut remarquer cette colonne contient seulement "GeoData Consultants Ltd" dupliqué dans tout les lignes


# on essaye avec une autre colonne
data.groupBy('water_quality').count().orderBy('count', ascending=False).show()

# on déduit que la colonne recroded_by n'est pas informative donc elle peut etre supprimé
data = data.drop('recorded_by')

# creation d'un pivot_table grouped by statut des puits dans chaque region et
# calculer la somme de la quantité totale d'eau dans les puits
# pivot_table() évite la répétition des données de la DataFrame .
# Elle résume les données et applique différentes fonctions d'agrégation sur les données.
data.groupBy('status_group').pivot('region').sum('amount_tsh').show()

# Convert categories with low frequency to Others.
# 1- afficher les string columns
print(str_cols)

# print les catégories actuelles et leur fréquence par ordre décroissant
# Stocker cette paire de catégories et de fréquences dans une variable.
# Itérer sur chaque paire et stocker le nom de la catégorie
# si sa fréquence est inférieure à 1000, dans une nouvelle variable appelée less_than

for column in str_cols[:3]:  # afficher les 3 premiers col de str_cols
    print(data.groupBy(column).count().orderBy('count', ascending=False).show())
    values_cat = data.groupBy(column).count().collect()

    # si une catégorie apparaît dans la variable less_than, remplacez la catégorie par une nouvelle catégorie appelée
    # "others"sinon, gardez la catégorie.
    lessthan = [x[0] for x in values_cat if x[1] < 1000]  # 1000 est arbitraire
    data = data.withColumn(column, when(col(column).isin(lessthan), 'Others').otherwise(col(column)))
    data.groupBy(column).count().orderBy('count', ascending=False).show()

    # calculer les valeur manquantes dans les numeric colonnes (plus precisemment la colonne population(la
    # poupulation autour de chaque puit) la fréquence de chaque population enregistrée et orderby la population
    # plutôt que par la fréquence
    data.groupBy('population').count().orderBy('population').show()

    # on peut remplacer cette population par la moyenne population dans un district où le puits est situé.
    # (----Pour ce faire, d'abord, imputez ces valeurs de population avec des valeurs nulles----)
    data = data.withColumn('population', when(col('population') < 2, lit(None)).otherwise(col('population')))

    # partition de la data sur la col "district code"
    w = Window.partitionBy(data['district_code'])

    # Remplacer  toutes les valeurs nulles de la colonne par la moyenne population sur la fenêtre de partition
    data = data.withColumn('population',
                           when(col('population').isNull(), avg(data['population']).over(w)).otherwise(
                               col('population')))

    # arrondir les valeursde population imputées  à l'aide de la fonction ceil() (renvoie la val absolue)
    data = data.withColumn('population', ceil(data['population']))

    # afficher les frequences de chque population enregistrée et classée par la population encore une fois
    data.groupBy('population').count().orderBy('population').show()

"""Spark n'a pas de bibliothèque de visualisation, il faut donc convertir le spark dataframe en un pandas dataframe 
avant de créer des visualisations, Comme le dataframe peut être extrêmement volumineux ,on peut sélectionner les 
colonnes adequates pour les graphs , puis convertir ce sous-ensemble de données en un Pandas dataframe """

color_status = {'functional': 'green', 'non functional': 'red', 'functional needs repair': 'blue'}

# choix des 5 colonnes puis convertir le sous_dataframe en un pandas data frame
cols = ['status_group', 'payment_type', 'longitude', 'latitude', 'gps_height']
df = data.select(cols).toPandas()
print(df)

# barplot du nombre de puits dans chaque 'payment_type'
fig, ax = plt.subplots(figsize=(25, 20))
sns.countplot(x='payment_type', hue='status_group', data=df, ax=ax, palette=color_status)
plt.xticks(rotation=45)
fig.savefig('result_images/show_nb_well_per_payment_type.png')

# la latitude et la longitude de chaque puits sous forme de nuage de points
fig1, ax1 = plt.subplots(12, 8)
sns.scatterplot(data=df, x='longitude', y='latitude', hue='status_group', ax=ax1, palette=color_status)
fig1.savefig('result_images/show_lay_long.png')

# histogramme avec des estimations de densité de noyau de la colonne GPS_height
# séparer la dataframe en trois sous-ensembles, un pour chaque type de statut
row_funtional = (df['status_group'] == 'functional')
row_non_funtional = (df['status_group'] == 'non functional')
row_repair = (df['status_group'] == 'functional needs repair')

# distplot pour chaque subset utilisant la colonne GPS_height et le meme code de couleur
col = 'gps_height'
fig2, ax2 = plt.subplots(12, 8)
sns.histplot(df[row_funtional][col], color='green', label='functional', ax=ax2)
sns.histplot(df[row_non_funtional][col], color='red', label='non functional', ax=ax2)
sns.histplot(df[row_repair][col], color='blue', label='functional needs repair', ax=ax2)
fig2.savefig('result_images/show_gps_height.png')

# Changement des valeurs nulles par 0 dans la colonne 'population'
data.na.fill(value=0, subset=["population"])

""" Nombre de puits par région et par statut"""
data_puits_region = data.select("status_group", "region")
data_puits_region_df = data_puits_region.toPandas()

color_status = {'functional': 'green', 'non functional': 'red', 'functional needs repair': 'blue'}
fig_well_region, ax = plt.subplots(figsize=(12, 8))
sns.countplot(x='region', hue='status_group', data=data_puits_region_df, ax=ax, palette=color_status)
plt.xticks(rotation=45)
fig_well_region.savefig('result_images/well_per_region.png')

""" Nombre de construction de puits par année """

cons_puit_per_year = data.groupBy("construction_year").agg(count("construction_year").alias("nb_construction"))
# Tri de la colonne 'nb_construction' par ordre croissant
cons_puit_per_year = cons_puit_per_year.sort("construction_year")
cons_puit_per_year.show()

# Transformation du type de la colonne 'construction_year' en string pour un meilleur affichage sur le graphe
cons_puit_per_year_str = cons_puit_per_year.withColumn('construction_year', col('construction_year').cast(StringType()))
# Converstion de la dataframe en dataframe Pandas
cons_puit_per_year_df = cons_puit_per_year_str.toPandas()
cons_puit_per_year_df.plot(x="construction_year", y="nb_construction", figsize=(15, 10))
plt.savefig('result_images/well_contruct_over_year.png')

""" Define the world map centered around Tanzania with a zoom level """

map_draw = folium.Map(location=[-3.852983, 36.756231], tiles='Stamen Toner', zoom_start=11.5)

# loop through the 100 crimes and add each to the incidents feature group
for index, row in df[df['longitude'].notnull()].iterrows():
    # element raduis
    elem_radius = round(((row['gps_height'] * 50) / max(df['gps_height'])), 2)

    # element opacity
    elem_opac = round((elem_radius / 50), 1)

    lat = row['latitude']
    longi = row['longitude']
    folium.CircleMarker(
        location=[lat, longi],
        color='#3186cc',
        fill=True,
        radius=elem_radius,
        fill_color='#3186cc',
        fill_opacity=elem_opac,
        popup="Status Group: " + str(row['status_group']) + "\nGPS Height: " + str(row['gps_height'])
    ).add_to(map_draw)

# display world map
map_draw.save("map_draw.html")

"""" Nombre total de puits par region """

data_region_puit = data.groupBy("region").agg(count("status_group").alias('nb_puits'))
data_region_puit.show()

""" Dataframe du nombre total d'habitants par région """
data_region_pop = data.groupBy("region").agg(sum("population").alias('population_total'))
data_region_pop.show()

""" Jointure de la dataframe du nombre total de population par région et celle du nombre total de puits par région 
Affichage du nombre total de puits et de populations par région """

data_reg_pop_puit = data_region_pop.join(data_region_puit, on="region")
data_reg_pop_puit.show()
data_reg_pop_puit_df = data_reg_pop_puit.toPandas()

data_reg_pop_puit_df.plot.bar(x='region', y=["population_total", "nb_puits"], rot=90, figsize=(15, 50))
plt.savefig('result_images/population_well_total_per_region.png')
