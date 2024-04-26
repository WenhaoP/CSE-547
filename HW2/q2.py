# Import the libraries we will need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import findspark
findspark.init()

from pyspark.sql import *
import pyspark.sql.functions as F
from pyspark.sql.types import *

def k_means(observation, initial_centroids, max_iter):
    def comb(x, y):
        if isinstance(x, list) & isinstance(y, list):
            return x + y
        elif isinstance(x, list):
            return x + [y]
        elif isinstance(y, list):
            return [x] + y
        else:
            return [x] + [y]
    
    def l2_distance(x, y):
        return float(np.linalg.norm(np.array(x) - np.array(y), ord=2) ** 2)
    l2_distance_udf = F.udf(l2_distance, FloatType())

    cost = np.zeros(max_iter + 1)

    # iteration 0 (initialization)
    new_partitions = observation.crossJoin(initial_centroids)
    new_partitions = new_partitions.withColumn("obs_to_c_dist", l2_distance_udf(F.col("observation"), F.col("centroid")))

    window = Window.partitionBy("obs_index")
    new_partitions = new_partitions.withColumn("min_dist", F.min("obs_to_c_dist").over(window))
    new_partitions = new_partitions.filter(((F.col("min_dist") - F.col("obs_to_c_dist")) < 1e-6) &
                                        ((F.col("obs_to_c_dist") - F.col("min_dist")) < 1e-6)).sort("obs_index").drop("min_dist")

    new_centroids = new_partitions.rdd.map(lambda row: (row["c_index"], row["observation"]))
    new_centroids = new_centroids.reduceByKey(lambda x, y: [comb(x[i], y[i]) for i in range(len(x))])
    new_centroids = new_centroids.map(lambda x: x if np.array(x[1]).ndim == 1 else (x[0], np.mean(np.array(x[1]), axis=1).tolist()))
    new_centroids = spark.createDataFrame(new_centroids).toDF("c_index", "centroid")

    cost[0] = np.sum(new_partitions.select("obs_to_c_dist").rdd.flatMap(lambda x: x).collect())
    print(f"At iteration 0, the cost is {cost[0]}.")

    # rest iterations
    for t in np.arange(1, max_iter + 1):
        new_partitions = observation.crossJoin(new_centroids)
        new_partitions = new_partitions.withColumn("obs_to_c_dist", l2_distance_udf(F.col("observation"), F.col("centroid")))

        window = Window.partitionBy("obs_index")
        new_partitions = new_partitions.withColumn("min_dist", F.min("obs_to_c_dist").over(window))
        new_partitions = new_partitions.filter(((F.col("min_dist") - F.col("obs_to_c_dist")) < 1e-6) &
                                            ((F.col("obs_to_c_dist") - F.col("min_dist")) < 1e-6)).sort("obs_index").drop("min_dist")

        new_centroids = new_partitions.rdd.map(lambda row: (row["c_index"], row["observation"]))
        new_centroids = new_centroids.reduceByKey(lambda x, y: [comb(x[i], y[i]) for i in range(len(x))])
        new_centroids = new_centroids.map(lambda x: x if np.array(x[1]).ndim == 1 else (x[0], np.mean(np.array(x[1]), axis=1).tolist()))
        new_centroids = spark.createDataFrame(new_centroids).toDF("c_index", "centroid")
        
        cost[t] = np.sum(new_partitions.select("obs_to_c_dist").rdd.flatMap(lambda x: x).collect())

        print(f"At iteration {t}, the cost is {cost[t]}.")

        new_centroids = new_centroids.collect()
        new_centroids = spark.createDataFrame(data=new_centroids)
    
    return cost, new_centroids.collect()

def k_medians(observation, initial_centroids, max_iter):
    def comb(x, y):
        if isinstance(x, list) & isinstance(y, list):
            return x + y
        elif isinstance(x, list):
            return x + [y]
        elif isinstance(y, list):
            return [x] + y
        else:
            return [x] + [y]
    
    def l1_distance(x, y):
        return float(np.linalg.norm(np.array(x) - np.array(y), ord=1))
    l1_distance_udf = F.udf(l1_distance, FloatType())

    cost = np.zeros(max_iter + 1)

    # iteration 0 (initialization)
    new_partitions = observation.crossJoin(initial_centroids)
    new_partitions = new_partitions.withColumn("obs_to_c_dist", l1_distance_udf(F.col("observation"), F.col("centroid")))

    window = Window.partitionBy("obs_index")
    new_partitions = new_partitions.withColumn("min_dist", F.min("obs_to_c_dist").over(window))
    new_partitions = new_partitions.filter(((F.col("min_dist") - F.col("obs_to_c_dist")) < 1e-6) &
                                        ((F.col("obs_to_c_dist") - F.col("min_dist")) < 1e-6)).sort("obs_index").drop("min_dist")

    new_centroids = new_partitions.rdd.map(lambda row: (row["c_index"], row["observation"]))
    new_centroids = new_centroids.reduceByKey(lambda x, y: [comb(x[i], y[i]) for i in range(len(x))])
    new_centroids = new_centroids.map(lambda x: x if np.array(x[1]).ndim == 1 else (x[0], np.median(np.array(x[1]), axis=1).tolist()))
    new_centroids = spark.createDataFrame(new_centroids).toDF("c_index", "centroid")

    cost[0] = np.sum(new_partitions.select("obs_to_c_dist").rdd.flatMap(lambda x: x).collect())
    print(f"At iteration 0, the cost is {cost[0]}.")

    # rest iterations
    for t in np.arange(1, max_iter + 1):
        new_partitions = observation.crossJoin(new_centroids)
        new_partitions = new_partitions.withColumn("obs_to_c_dist", l1_distance_udf(F.col("observation"), F.col("centroid")))

        window = Window.partitionBy("obs_index")
        new_partitions = new_partitions.withColumn("min_dist", F.min("obs_to_c_dist").over(window))
        new_partitions = new_partitions.filter(((F.col("min_dist") - F.col("obs_to_c_dist")) < 1e-6) &
                                            ((F.col("obs_to_c_dist") - F.col("min_dist")) < 1e-6)).sort("obs_index").drop("min_dist")

        new_centroids = new_partitions.rdd.map(lambda row: (row["c_index"], row["observation"]))
        new_centroids = new_centroids.reduceByKey(lambda x, y: [comb(x[i], y[i]) for i in range(len(x))])
        new_centroids = new_centroids.map(lambda x: x if np.array(x[1]).ndim == 1 else (x[0], np.median(np.array(x[1]), axis=1).tolist()))
        new_centroids = spark.createDataFrame(new_centroids).toDF("c_index", "centroid")

        cost[t] = np.sum(new_partitions.select("obs_to_c_dist").rdd.flatMap(lambda x: x).collect())

        print(f"At iteration {t}, the cost is {cost[t]}.")

        new_centroids = new_centroids.collect()
        new_centroids = spark.createDataFrame(data=new_centroids)
    
    return cost, new_centroids.collect()


def main():

    # create the Spark Session
    spark = SparkSession.builder.appName("Q2").getOrCreate()

    # loading and preprocessing the data
    data = spark.read.csv("hw2-bundle/hw2-bundle/kmeans/data/data.txt").toDF("observation")
    data = data.withColumn("observation", F.split(F.col("observation"), " ").cast("array<float>"))
    data = data.withColumn("obs_index", F.monotonically_increasing_id())
    window = Window.orderBy(F.col("obs_index"))
    data = data.withColumn("obs_index", F.row_number().over(window) - 1)

    init_C_random = spark.read.csv("hw2-bundle/hw2-bundle/kmeans/data/c1.txt").toDF("centroid")
    init_C_random = init_C_random.withColumn("centroid", F.split(F.col("centroid"), " ").cast("array<float>"))
    init_C_random = init_C_random.withColumn("c_index", F.monotonically_increasing_id())
    window = Window.orderBy(F.col("c_index"))
    init_C_random = init_C_random.withColumn("c_index", F.row_number().over(window) - 1)

    init_C_far = spark.read.csv("hw2-bundle/hw2-bundle/kmeans/data/c2.txt").toDF("centroid")
    init_C_far = init_C_far.withColumn("centroid", F.split(F.col("centroid"), " ").cast("array<float>"))
    init_C_far = init_C_far.withColumn("c_index", F.monotonically_increasing_id())
    window = Window.orderBy(F.col("c_index"))
    init_C_far = init_C_far.withColumn("c_index", F.row_number().over(window) - 1)

    # k-means clustering
    random_init_k_means_result = k_means(data, init_C_random, 20)
    far_init_k_means_result = k_means(data, init_C_far, 20)

    fig, ax = plt.subplots()
    ax.scatter(np.arange(0, 21), random_init_k_means_result[0], label="random")
    ax.scatter(np.arange(0, 21), far_init_k_means_result[0], label="far")
    ax.set_xlabel("# of iterations")
    ax.set_ylabel("cost")
    ax.set_title("k-means clustering")
    ax.legend()
    plt.show()

    print(f"The cost decrease by {1 - random_init_k_means_result[0][10] / random_init_k_means_result[0][0]} after 10 iterations for random initialization")
    print(f"The cost decrease by {1 - far_init_k_means_result[0][10] / far_init_k_means_result[0][0]} after 10 iterations for far initialization")

    # k-medians clustering
    random_init_k_medians_result = k_medians(data, init_C_random, 20)
    far_init_k_medians_result = k_medians(data, init_C_far, 20)

    fig, ax = plt.subplots()
    ax.scatter(np.arange(0, 21), random_init_k_medians_result[0], label="random")
    ax.scatter(np.arange(0, 21), far_init_k_medians_result[0], label="far")
    ax.set_xlabel("# of iterations")
    ax.set_ylabel("cost")
    ax.set_title("k-medians clustering")
    ax.legend()
    plt.show()

    print(f"The cost decrease by {1 - random_init_k_medians_result[0][10] / random_init_k_medians_result[0][0]} after 10 iterations for random initialization")
    print(f"The cost decrease by {1 - far_init_k_medians_result[0][10] / far_init_k_medians_result[0][0]} after 10 iterations for far initialization")

    spark.stop()

if __name__ == "__main__":
    main()