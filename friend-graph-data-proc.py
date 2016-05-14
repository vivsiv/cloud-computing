#TO RUN open shell with command:pyspark --packages graphframes:graphframes:0.1.0-spark1.6
from pyspark.sql import SQLContext
from graphframes import *

sqlContext = SQLContext(sc)

#load user data
users = sqlContext.read.json("data/yelp_academic_dataset_user.json")
#552339

#load business data
business = sqlContext.read.json("data/yelp_academic_dataset_business.json")
#get just Vegas businesses
lv_business = business.filter(business_df["city"] == "Las Vegas")
#get just vegas restaurants
lv_restaurants = lv_business.select("*").where("array_contains(categories,'Restaurants')")
#load review data
review = sqlContext.read.json("data/yelp_academic_dataset_review.json")
#get user ids who have reviewed lv_businesses
lv_user_ids = review.join(lv_restaurants.select("business_id"),"business_id").groupBy("user_id").count().select("user_id")
#compute vertices
vertices = lv_user_ids.withColumnRenamed("user_id","id")
#vertices.count()
#174479

#compute edges
user_friends_pairs = users.map(lambda user: (user.user_id,user.friends))
user_friend_edges = user_friends_pairs.flatMapValues(lambda user: user).toDF(['user_id','friend_id'])
edges = user_friend_edges.withColumnRenamed("user_id","src").withColumnRenamed("friend_id","dst")

friend_graph = GraphFrame(vertices, edges)

# print friend_graph

# friend_graph.vertices.show()
# friend_graph.edges.show()

#cc = friend_graph.connectedComponents().vertices

