#TO RUN open shell with command:pyspark --packages graphframes:graphframes:0.1.0-spark1.6
from pyspark.sql import SQLContext
from graphframes import *

#load user data
users = sqlContext.read.json("data/yelp_academic_dataset_user.json")

#compute vertices
vertices = users.select("user_id").withColumnRenamed("user_id","id")

#compute edges
user_friends_pairs = users.map(lambda user: (user.user_id,user.friends))
user_friend_edges = user_friends_pairs.flatMapValues(lambda user: user).toDF(['user_id','friend_id'])
edges = user_friend_edges.withColumnRenamed("user_id","src").withColumnRenamed("friend_id","dst")

friend_graph = GraphFrame(vertices, edges)

# print friend_graph

# friend_graph.vertices.show()
# friend_graph.edges.show()

#cc = friend_graph.connectedComponents().vertices

