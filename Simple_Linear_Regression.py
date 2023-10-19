from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType
from pyspark.sql.functions import col
import sys
import time

spark = SparkSession.builder.appName("TaxiDataAnalysis").getOrCreate()

df = spark.read.csv(sys.argv[1], header=False, inferSchema=True)

def is_float(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

# Register the UDF with PySpark
is_float_udf = udf(is_float, BooleanType())

# Define conditions using the UDF and PySpark functions
corrected_df = df.filter(is_float_udf("_c5") & is_float_udf("_c11"))

# Show the cleaned DataFrame
corrected_df.show()

columns = ["medallion", "hack_license", "pickup_datetime", "dropoff_datetime", "trip_time_in_secs", "trip_distance", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "payment_type", "fare_amount", "surcharge", "mta_tax", "tip_amount", "tolls_amount", "total_amount"]
renamed_columns = ["_c0", "_c1", "_c2", "_c3","_c4", "_c5", "_c6", "_c7","_c8", "_c9", "_c10", "_c11","_c12", "_c13", "_c14", "_c15","_c16"]

for orig_col, new_col in zip(renamed_columns, columns):
  corrected_df = corrected_df.withColumnRenamed(orig_col, new_col)

corrected_df.show()

corrected_df = corrected_df.withColumn("pickup_datetime", corrected_df["pickup_datetime"].cast("timestamp"))
corrected_df = corrected_df.withColumn("dropoff_datetime", corrected_df["dropoff_datetime"].cast("timestamp"))

corrected_df = corrected_df.filter(
    (col("trip_distance") >= 1) & (col("trip_distance") <= 50) &
    (col("fare_amount") >= 3) & (col("fare_amount") <= 200) &
    (col("tolls_amount") >= 3) &
    ((col("dropoff_datetime").cast("long") - col("pickup_datetime").cast("long")) >= 120) &
    ((col("dropoff_datetime").cast("long") - col("pickup_datetime").cast("long")) <= 3600)
)

# Show the filtered DataFrame
corrected_df.show()    

import numpy as np
from scipy import stats

# Extract "trip_distance" and "fare_amount" columns as numpy arrays
trip_distance = np.array(corrected_df.select("trip_distance").rdd.flatMap(lambda x: x).collect(), dtype=float)
fare_amount = np.array(corrected_df.select("fare_amount").rdd.flatMap(lambda x: x).collect(), dtype=float)

slope, intercept, r_value, p_value, std_err = stats.linregress(trip_distance, fare_amount)

print("Slope (m):", slope)
print("Intercept (b):", intercept)

start_time = time.time()  # Record start time
computation_time = time.time() - start_time
print("Computation Time:", computation_time, "seconds")

spark.stop()