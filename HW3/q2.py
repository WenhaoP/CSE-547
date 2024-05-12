# Import the libraries we will need
import pandas as pd
import numpy as np

import findspark
findspark.init()

from pyspark.sql import *
import pyspark.sql.functions as F
from pyspark.sql.types import *

def main():
    # create the Spark Session
    spark = SparkSession.builder.appName("Q2").getOrCreate()

    ### Page Rank ###
    full = spark.sparkContext.textFile("hw3-bundle/hw3-bundle/pagerank_hits/data/graph-full.txt")
    full = full.map(lambda x: (int(x.split()[0]), (int(x.split()[1])))).distinct() # (source, destination)
    out_deg = full.map(lambda x: (x[0], 1)).reduceByKey(lambda v1, v2: v1 + v2).map(lambda x: (x[0], 1 / x[1]))
    M = full.join(out_deg).map(lambda x: (x[1][0], x[0], x[1][1]))

    beta = 0.8
    n = 1000
    r = np.ones(n) / n

    for i in range(40):
        M_dot_r = M.map(lambda x: (x[0], x[2] * r[int(x[1] - 1)])).reduceByKey(lambda v1, v2: v1 + v2).sortByKey()
        M_dot_r_np = np.array(M_dot_r.map(lambda x: x[1]).collect())
        r = beta * M_dot_r_np + (1 - beta) / n

        print(f"At iteration {i}, top 5 scores are {np.sort(r)[-5:]}")
    
    # top 5
    print(f"top 5 node ids and scores are {np.argpartition(r, -5)[-5:] + 1} and {np.partition(r, -5)[-5:]}")

    # bottom 5
    print(f"bottom 5 node ids and scores are {np.argpartition(r, 5)[:5] + 1} and {np.partition(r, 5)[:5]}")


    ### HITS ###

    full = spark.sparkContext.textFile("hw3-bundle/hw3-bundle/pagerank_hits/data/graph-full.txt")
    L = full.map(lambda x: (int(x.split()[0]), (int(x.split()[1])))).distinct() # (source, destination)
    L_T = L.map(lambda x: (x[1], x[0]))

    n = 1000
    lam = 1
    mu = 1

    h = np.ones(n)

    for i in range(40):
        L_T_h = L_T.map(lambda x: (x[0], h[int(x[1] - 1)])).reduceByKey(lambda v1, v2: v1 + v2).sortByKey()
        a = np.array(L_T_h.map(lambda x: x[1]).collect())
        a = a / a.max()

        L_a = L.map(lambda x: (x[0], a[int(x[1] - 1)])).reduceByKey(lambda v1, v2: v1 + v2).sortByKey()
        h = np.array(L_a.map(lambda x: x[1]).collect())
        h = h / h.max()

        print(f"At iteration {i}, top 5 hubbiness scored nodes are {np.argpartition(h, -5)[-5:] + 1}")
        print(f"At iteration {i}, top 5 hubbiness scores are {np.partition(h, -5)[-5:]}")
        print(f"At iteration {i}, bottom 5 hubbiness scored nodes are {np.argpartition(h, 5)[:5] + 1}")
        print(f"At iteration {i}, bottom 5 hubbiness scores are {np.partition(h, 5)[:5]}")

        print(f"At iteration {i}, top 5 authority scored nodes are {np.argpartition(a, -5)[-5:] + 1}")
        print(f"At iteration {i}, top 5 authority scores are {np.partition(a, -5)[-5:]}")
        print(f"At iteration {i}, bottom 5 authority scored nodes are {np.argpartition(a, 5)[:5] + 1}")
        print(f"At iteration {i}, bottom 5 authority scores are {np.partition(a, 5)[:5]}")

    # top 5
    print(f"top 5 node ids and hubbiness scores are {np.argpartition(h, -5)[-5:] + 1} and {np.partition(h, -5)[-5:]}")
    print(f"top 5 node ids and authority scores are {np.argpartition(a, -5)[-5:] + 1} and {np.partition(a, -5)[-5:]}")

    # bottom 5
    print(f"bottom 5 node ids and hubbiness scores are {np.argpartition(h, 5)[:5] + 1} and {np.partition(h, 5)[:5]}")
    print(f"bottom 5 node ids and authority scores are {np.argpartition(a, 5)[:5] + 1} and {np.partition(a, 5)[:5]}")

    spark.stop()

if __name__ == "__main__":
    main()