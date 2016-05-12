from pyspark.sql import SQLContext

#load user data
users = sqlContext.read.json("data/yelp_academic_dataset_user.json")

#compute vertices
vertices = users.select("user_id").withColumnRenamed("user_id","id")

#compute edges
user_friends_pairs = users.map(lambda user: (user.user_id,user.friends))
user_friend_edges = user_friends_pairs.flatMapValues(lambda user: user).toDF(['user_id','friend_id'])
edges = user_friend_edges.withColumnRenamed("user_id","src").withColumnRenamed("friend_id","dst")