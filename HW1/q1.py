# Import the libraries we will need
import pandas as pd
import numpy as np

import findspark
findspark.init()

import pyspark
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
import pyspark.pandas as ps

def main():

    # create the Spark Session
    spark = SparkSession.builder.appName("Q1").getOrCreate()

    # read and preprocess the data
    friend = spark.read.csv("hw1-bundle/hw1-bundle/q1/data/soc-LiveJournal1Adj.txt", sep='\t')
    friend = friend.toDF("User", "Friends")
    friend = friend.withColumn("User", friend["User"].cast("int"))
    friend = friend.withColumn("Friends", F.split(friend["Friends"], ",").cast("array<int>"))

    # filter the users with no friends
    no_friend = friend.filter(F.size(friend["Friends"]) == -1).withColumnRenamed("Friends", "Recommendations") 

    # filter the users with friends
    friend = friend.filter(F.size(friend["Friends"]) != -1)

    # find the users who are not friends with each user
    all_user_ids = friend.select("User").rdd.flatMap(lambda x: x).collect()
    all_user_ids = spark.createDataFrame([(all_user_ids,)], ["All Users"])
    friend = friend.crossJoin(all_user_ids)
    unfriend = friend.withColumn("Unfriends", F.array_except(friend["All Users"], friend["Friends"])).select("User", "Unfriends")
    unfriend = unfriend.withColumn("Unfriend", F.explode(unfriend["Unfriends"]))
    unfriend = unfriend.withColumn("Unfriend", unfriend["Unfriend"].cast("int"))
    unfriend = unfriend.filter(unfriend["User"] != unfriend["Unfriend"])
    friend = friend.select(["User", "Friends"])

    # find the mutual friends 
    mutual_friend = unfriend.join(friend, on="User", how="left").withColumnRenamed("Friends", "User's Friends")
    mutual_friend = mutual_friend.join(friend.withColumnRenamed("User", "Unfriend"), on="Unfriend", how="left").withColumnRenamed("Friends", "Unfriend's Friends")
    mutual_friend = mutual_friend.withColumn("Mutual Friends", F.array_intersect(mutual_friend["User's Friends"], mutual_friend["Unfriend's Friends"]))
    mutual_friend = mutual_friend.withColumn("Num of Mutual Friends", F.size(mutual_friend["Mutual Friends"]))
    mutual_friend = mutual_friend.select(["User", "Unfriend", "Num of Mutual Friends"])

    # find the recommendations
    window_spec = Window.partitionBy("User").orderBy(F.desc("Num of Mutual Friends"), F.asc("Unfriend"))
    mutual_friend = mutual_friend.withColumn("rank", F.row_number().over(window_spec))
    mutual_friend = mutual_friend.filter(mutual_friend["rank"] <= 10)
    mutual_friend = mutual_friend.select(["User", "Unfriend"]).groupby("User").agg(F.collect_list("Unfriend").alias("Recommendations"))
    final = mutual_friend.union(no_friend).sort(F.asc("user"))
    print(final.filter((final["User"] == 924) | 
             (final["User"] == 8941) | 
             (final["User"] == 8942) |
             (final["User"] == 9019) |
             (final["User"] == 9020) |
             (final["User"] == 9021) | 
             (final["User"] == 9022) | 
             (final["User"] == 9990) |
             (final["User"] == 9992) |
             (final["User"] == 9993)).take(10))
    
    spark.stop()

if __name__ == "__main__":
    main()