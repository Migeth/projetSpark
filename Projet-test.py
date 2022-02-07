from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import isnan, when, count, col, lit, trim, avg, ceil
from pyspark.sql.types import StringType
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# import wget



#tapper ces commandes sous le terminal de pycharm pour telecharger directement les fichiers a partir de leur URL
#-------  wget https://s3.amazonaws.com/drivendata/data/7/public/4910797b-ee55-40a7-8668-10efd5c1b960.csv -O features.csv
#le 2eme data set contient les étiquettes décrivant l'état des puits

# tapper ces commandes sous le terminal de pycharm pour telecharger directement les fichiers a partir de leur URL
# -------  wget https://s3.amazonaws.com/drivendata/data/7/public/4910797b-ee55-40a7-8668-10efd5c1b960.csv -O features.csv
# le 2eme data set contient les étiquettes décrivant l'état des puits

# ---------- wget https://s3.amazonaws.com/drivendata/data/7/public/0bf8bc6e-30d0-4c50-956a-603fc693d966.csv -O labels.csv

# creation d'une spark session
sc = SparkSession.builder.master("local[*]").getOrCreate()

feature = sc.read.csv("features.csv", inferSchema=True, header=True)
label = sc.read.csv("labels.csv", inferSchema=True, header=True)

print(feature.count())
print(label.count())
feature.printSchema()
label.printSchema()

# Fusionner les deux dataset
data = feature.join(label, on=("id"))
print(data.count())
data.printSchema()

# Affichage des 10 premieres lignes du dataframe
data.show(n=10)


# La fonction withColumn prend deux arguments -le nom de la nouvelle colonne et l'expression de la nouvelle colonne

data = data.withColumn('region_code', col('region_code').cast(StringType())).withColumn('district_code', col('district_code').cast(StringType()))
data.printSchema()

#supprimer les doublons
data=data.dropDuplicates(["id"])
data.count()

#en utilisant la fonction trim(), on peut supprimer les espaces blancs dans chaque de ces colonnes.

str_cols = [item[0] for item in data.dtypes if item[1].startswith('string')]

for cols in str_cols:

  data= data.withColumn(cols, trim(data[cols]))


#supprimer les colonnes avec les valeurs nulles a partir d'un certain seuil

data.select([count(when(isnan(c) | col(c).isNull(),c  )).alias(c) for c in data.columns if c not in {'date_recorded', 'public_meeting','permit' } ]).show()

agg_row= data.select([(count(when(isnan(c) | col(c).isNull(),c  ))/data.count()).alias(c) for c in data.columns if c not in {'date_recorded', 'public_meeting','permit' } ]).collect()
#convertir le résultat dans un dictionnaire pour une itération facile

agg_dict_list =[row.asDict() for row in agg_row]

agg_dict= agg_dict_list[0]

#on itére sur ce dictionnaire et stocke les noms de colonnes, qui ont nombre moyen de valeurs nulles plus qu'un seuil donné

#on a choisit 0.4 par rapport aux 40%

col_null = list({i for i in agg_dict if agg_dict[i] > 0.4 })

print(agg_dict)

print(col_null)

data = data.drop(*col_null)

##groupping et aggregation
#count utilise l'ordre asc
data.groupBy('recorded_by').count().show()
#comme on peut remarquer cette colonne contient seulement "GeoData Consultants Ltd" dupliqué dans tout les lignes

#on essaye avec une autre colonne
data.groupBy('water_quality').count().orderBy('count', ascending= False).show()

#on déduit que la colonne recroded_by n'est pas informative donc elle peut etre supprimé
data = data.drop('recorded_by')

# creation d'un pivot_table grouped by statut des puits dans chaque region et
# calculer la somme de la quantité totale d'eau dans les puits
#pivot_table() évite la répétition des données de la DataFrame .
# Elle résume les données et applique différentes fonctions d'agrégation sur les données.
data.groupBy('status_group').pivot('region').sum('amount_tsh').show()














