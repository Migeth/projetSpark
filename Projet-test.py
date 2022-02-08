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




#Convert categories with low frequency to Others.
#1- afficher les string columns
print(str_cols)


#print les catégories actuelles et leur fréquence par ordre décroissant

#Stocker cette paire de catégories et de fréquences dans une variable.

#Itérer sur chaque paire et stocker le nom de la catégorie

#si sa fréquence est inférieure à 1000, dans une nouvelle variable appelée less_than



for column in str_cols [:3]: #afficher les 3 premiers col de str_cols

  print(data.groupBy(column).count().orderBy('count', ascending= False).show())

  values_cat = data.groupBy(column).count().collect()

  #si une catégorie apparaît dans la variable less_than, remplacez la catégorie par une nouvelle catégorie appelée "others"sinon, gardez la catégorie.

  lessthan = [x[0] for x in values_cat if x[1]< 1000] #1000 est arbitraire

  data= data.withColumn(column, when(col(column).isin(lessthan), 'Others').otherwise(col(column)))

  data.groupBy(column).count().orderBy('count', ascending= False).show()

  # calculer les valeur manquantes dans les numeric colonnes (plus precisemment la colonne population(la poupulation autour de chaque puit)
  # la fréquence de chaque population enregistrée et orderby la population plutôt que par la fréquence

  data.groupBy('population').count().orderBy('population').show()

  #on peut remplacer cette population par la moyenne population dans un district où le puits est situé.
  #(----Pour ce faire, d'abord, imputez ces valeurs de population avec des valeurs nulles----)

  data = data.withColumn('population', when(col('population') < 2, lit(None)).otherwise(col('population')))

  # partition de la data sur la col "district code"

  w = Window.partitionBy(data['district_code'])

  # Remplacer  toutes les valeurs nulles de la colonne par la moyenne population sur la fenêtre de partition

  data = data.withColumn('population',
                         when(col('population').isNull(), avg(data['population']).over(w)).otherwise(col('population')))

  # arrondir les valeursde population imputées  à l'aide de la fonction ceil() (renvoie la val absolue)

  data = data.withColumn('population', ceil(data['population']))

  # afficher les frequences de chque population enregistrée et classée par la population encore une fois

  data.groupBy('population').count().orderBy('population').show()












