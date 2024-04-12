# Import the libraries we will need
import pandas as pd
import numpy as np
import itertools

import findspark
findspark.init()

import pyspark
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
import pyspark.pandas as ps

def main():

    # create the Spark Session
    spark = SparkSession.builder.appName("Q2").getOrCreate()

    s = 100 # support threshold

    # read and preprocess the data
    browsing = spark.read.csv("hw1-bundle/hw1-bundle/q2/data/browsing.txt", sep='\t')
    browsing = browsing.toDF("Items")
    browsing = browsing.withColumn("Items", F.split("Items", " ").cast("array<string>"))

    # find the frequent items
    browsing_exploded = browsing.withColumn("Item", F.explode(browsing["Items"]))
    browsing_exploded = browsing_exploded.filter(F.length("Item") == 8)
    browsing_grouped = browsing_exploded.groupBy("Item").count().withColumnRenamed("count", "Frequency")
    freq_items = browsing_grouped.filter(F.col("Frequency") >= s)

    # find the frequent pairs
    frequent_pairs = browsing.withColumn("Frequent_items", F.lit(freq_items.select("Item").rdd.flatMap(lambda x: x).collect()))
    frequent_pairs = frequent_pairs.withColumn("Frequent_items", F.array_intersect("Items", "Frequent_items"))
    frequent_pairs = frequent_pairs.withColumn("Frequent_pairs", (F.udf(lambda x: list(itertools.combinations(x, 2)), "array<array<string>>"))(F.col("Frequent_items")))
    frequent_pairs = frequent_pairs.withColumn("Frequent_pair", F.explode("Frequent_pairs")).select("Frequent_pair")
    frequent_pairs = frequent_pairs.withColumn("Frequent_pair", F.sort_array("Frequent_pair"))
    frequent_pairs = frequent_pairs.groupBy("Frequent_pair").count().withColumnRenamed("count", "Frequency")
    frequent_pairs = frequent_pairs.filter(F.col("Frequency") >= s).sort([F.desc("Frequency"), F.asc("Frequent_pair")])

    print("Top 5 Frequent Pairs:\n")
    frequent_pairs.show(5)

    # generate the association rules from the frequent pairs and compute their confidence scores
    frequent_pairs = frequent_pairs.withColumns({
        "X": (F.udf(lambda x: x[0]))(F.col("Frequent_pair")),
        "Y": (F.udf(lambda x: x[1]))(F.col("Frequent_pair")),
    })
    frequent_pairs = frequent_pairs.join(freq_items.withColumnRenamed("Frequency", "X_frequency"), frequent_pairs["X"] == freq_items["Item"], "left").drop("Item")
    frequent_pairs = frequent_pairs.join(freq_items.withColumnRenamed("Frequency", "Y_frequency"), frequent_pairs["Y"] == freq_items["Item"], "left").drop("Item")
    frequent_pairs = frequent_pairs.withColumns({
        "Forward_conf": F.col("Frequency") / F.col("X_frequency"),
        "Backward_conf": F.col("Frequency") / F.col("Y_frequency")
    })
    forward_rules = frequent_pairs.select(["X", "Y", "Forward_conf"]).withColumnsRenamed({
        "X": "Left",
        "Y": "Right",
        "Forward_conf": "Confidence"
    })
    backward_rules = frequent_pairs.select(["Y", "X", "Backward_conf"]).withColumnsRenamed({
        "Y": "Left",
        "X": "Right", 
        "Backward_conf": "Confidence"})
    frequent_pairs = frequent_pairs.select(["Frequent_pair", "Frequency"])
    pair_rules = forward_rules.union(backward_rules)
    pair_rules = pair_rules.sort([F.desc("Confidence"), F.asc("Left")])

    print("Top 5 Confident Rules:\n")
    pair_rules.show(5, False)

    # find the frequent triples
    frequent_triples = browsing.withColumn("Frequent_items", F.lit(freq_items.select("Item").rdd.flatMap(lambda x: x).collect()))
    frequent_triples = frequent_triples.withColumn("Frequent_items", F.array_intersect("Items", "Frequent_items"))
    frequent_triples = frequent_triples.withColumn("Frequent_triples", (F.udf(lambda x: list(itertools.combinations(x, 3)), "array<array<string>>"))(F.col("Frequent_items")))
    frequent_triples = frequent_triples.withColumn("Frequent_triple", F.explode("Frequent_triples")).select("Frequent_triple")

    frequent_triples = frequent_triples.withColumn("Frequent_triple", F.sort_array("Frequent_triple"))
    frequent_triples = frequent_triples.groupBy("Frequent_triple").count().withColumnRenamed("count", "Frequency")
    frequent_triples = frequent_triples.filter(F.col("Frequency") >= s).sort([F.desc("Frequency"), F.asc("Frequent_triple")])

    print("Top 5 Frequent Triples:\n")
    frequent_triples.show(5, False)

    # generate the association rules from the frequent triples and compute their confidence scores
    frequent_triples = frequent_triples.withColumns({
        "X": (F.udf(lambda x: x[0], "string"))(F.col("Frequent_triple")),
        "Y": (F.udf(lambda x: x[1], "string"))(F.col("Frequent_triple")),
        "Z": (F.udf(lambda x: x[2], "string"))(F.col("Frequent_triple")),
        "X_Y": (F.udf(lambda x: [x[0]] + [x[1]], "array<string>"))(F.col("Frequent_triple")),
        "X_Z": (F.udf(lambda x: [x[0]] + [x[2]], "array<string>"))(F.col("Frequent_triple")),
        "Y_Z": (F.udf(lambda x: [x[1]] + [x[2]], "array<string>"))(F.col("Frequent_triple")),
    })

    frequent_triples = frequent_triples.join(frequent_pairs.withColumnRenamed("Frequency", "X_Y_frequency"), frequent_triples["X_Y"] == frequent_pairs["Frequent_pair"], "left").drop("Frequent_pair")
    frequent_triples = frequent_triples.join(frequent_pairs.withColumnRenamed("Frequency", "X_Z_frequency"), frequent_triples["X_Z"] == frequent_pairs["Frequent_pair"], "left").drop("Frequent_pair")
    frequent_triples = frequent_triples.join(frequent_pairs.withColumnRenamed("Frequency", "Y_Z_frequency"), frequent_triples["Y_Z"] == frequent_pairs["Frequent_pair"], "left").drop("Frequent_pair")

    frequent_triples = frequent_triples.withColumns({
        "X_Y_to_Z_conf": F.col("Frequency") / F.col("X_Y_frequency"),
        "X_Z_to_Y_conf": F.col("Frequency") / F.col("X_Z_frequency"),
        "Y_Z_to_X_conf": F.col("Frequency") / F.col("Y_Z_frequency"),
    })

    X_Y_to_Z_rules = frequent_triples.select(["X_Y", "Z", "X_Y_to_Z_conf"]).withColumnsRenamed({
        "X_Y": "Left",
        "Z": "Right",
        "X_Y_to_Z_conf": "Confidence"
    })
    X_Z_to_Y_rules = frequent_triples.select(["X_Z", "Y", "X_Z_to_Y_conf"]).withColumnsRenamed({
        "X_Z": "Left",
        "Y": "Right",
        "X_Z_to_Y_conf": "Confidence"
    })
    Y_Z_to_X_rules = frequent_triples.select(["Y_Z", "X", "Y_Z_to_X_conf"]).withColumnsRenamed({
        "Y_Z": "Left",
        "X": "Right",
        "Y_Z_to_X_conf": "Confidence"
    })
    triple_rules = X_Y_to_Z_rules.union(X_Z_to_Y_rules).union(Y_Z_to_X_rules)
    triple_rules = triple_rules.sort([F.desc("Confidence"), F.asc("Left")])
    print("Top 5 Confident Rules:\n")
    triple_rules.show(5, False)
    
    spark.stop()

if __name__ == "__main__":
    main()