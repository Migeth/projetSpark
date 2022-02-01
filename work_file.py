import pyspark
from pyspark import SparkContext
from pyspark import SparkFiles
from pyspark.sql import SparkSession


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

