from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import isnan, when, count, col, lit, trim, avg, ceil
from pyspark.sql.types import StringType
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#import wget


#tapper ces commandes sous le terminal de pycharm pour telecharger directement les fichiers a partir de leur URL
#-------  wget https://s3.amazonaws.com/drivendata/data/7/public/4910797b-ee55-40a7-8668-10efd5c1b960.csv -O features.csv
#le 2eme data set contient les étiquettes décrivant l'état des puits
# ---------- wget https://s3.amazonaws.com/drivendata/data/7/public/0bf8bc6e-30d0-4c50-956a-603fc693d966.csv -O labels.csv

#creation d'une spark session
sc = SparkSession.builder.master("local[*]").getOrCreate()

feature = sc.read.csv("features.csv", inferSchema=True, header=True)
label = sc.read.csv("labels.csv", inferSchema=True, header=True)

print(feature.count())
print(label.count())
feature.printSchema()
label.printSchema()

 #Fusionner les deux dataset
data = feature.join(label, on = ("id") )
print(data.count())
data.printSchema()


#Affichage des 10 premieres lignes du dataframe
data.show(n=10)

#visualisation
color_status = {'functional': 'green', 'non functional': 'red', 'functional needs repair': 'blue'}

#creer un data

cols = ['status_group', 'payment_type', 'longitude', 'latitude', 'gps_height']
df= data.select(cols).toPandas()