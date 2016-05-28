#TO RUN open shell with command:pyspark --packages graphframes:graphframes:0.1.0-spark1.6
from pyspark.sql import SQLContext
from graphframes import *
import random
from pyspark.sql.functions import udf
from pyspark.sql.functions import lit
from pyspark.sql.types import *
from pyspark.sql.functions import col

sqlContext = SQLContext(sc)

#load data
users = sqlContext.read.json("data/yelp_academic_dataset_user.json")
#552339
business = sqlContext.read.json("data/yelp_academic_dataset_business.json")
review = sqlContext.read.json("data/yelp_academic_dataset_review.json")

#get just Vegas restaurants
lv_business = business.filter(business["city"] == "Las Vegas")
lv_restaurants = lv_business.select("*").where("array_contains(categories,'Restaurants')")

#get user ids who have reviewed lv_businesses
lv_user_ids = review.join(lv_restaurants.select("business_id"),"business_id").groupBy("user_id").count().select("user_id")

lv_reviews = review.join(lv_restaurants.select("business_id"),"business_id").select("business_id","user_id","stars","date")



### FRIEND GRAPH ###
#compute vertices
friend_vertices = lv_user_ids.withColumnRenamed("user_id","id")
#friend_vertices.count()
#174479

#compute friend_edges
user_friends_pairs = users.map(lambda user: (user.user_id,user.friends))
user_friend_edges = user_friends_pairs.flatMapValues(lambda user: user).toDF(['user_id','friend_id'])
friend_edges = user_friend_edges.withColumnRenamed("user_id","src").withColumnRenamed("friend_id","dst")

friend_graph = GraphFrame(friend_vertices, friend_edges)


### BUSINESS GRAPH ###
#generate lv_restaurant vertices set
lv_vertices = lv_restaurants.select("business_id").withColumnRenamed("business_id","id").withColumn("type",lit("business"))
#generate business graph user vertices
users_vertices = users.select("user_id").withColumnRenamed("user_id","id").withColumn("type",lit("user"))
business_vertices = users_vertices.unionAll(lv_vertices)

#load review data
# lv_reviews = review.join(lv_restaurants.select("business_id"),"business_id").select("business_id","user_id","stars","date")

#user-business graph edges
business_edges = lv_reviews.withColumnRenamed("user_id","src").withColumnRenamed("business_id","dst")
business_graph = GraphFrame(business_vertices, business_edges)

### MOST INFLUENTIAL USERS LINK PREDICTION ###
num_friends = friend_graph.inDegrees
most_pop_users = num_friends.sort(col("inDegree").desc()).limit(100)
most_pop_rest = lv_reviews.join(most_pop_users.select("id"), most_pop_users.id == lv_reviews.user_id).select("business_id","user_id","stars","date")
divide_point = int(round(most_pop_rest.count() * 0.5))
most_pop_rest1 = sqlContext.createDataFrame(most_pop_rest.orderBy("date").collect()[:divide_point])

most_pop_rest1_arr = most_pop_rest1.select("business_id").map(lambda r: r.business_id).collect()
def addPrediction(arg): return random.choice(most_pop_rest1_arr)
predictionAdder = udf(addPrediction, StringType())
business_graph_new = business_graph.edges.withColumn("predict_dst", predictionAdder(business_graph.edges['src']))
business_graph_predictions = business_graph_new.select("src","predict_dst").withColumnRenamed("predict_dst","dst")


### SOCIAL NETWORK LINK PREDICTION ###
# user_friends_pairs_df = users.map(lambda user: (user.user_id,user.friends)).toDF(['user_id','friends'])
# predictions = {}
# users_to_predict = user_friends_pairs_df.limit(10).collect()
# for user in users_to_predict:
# 	restaurants = []
# 	for friend in user.friends:
# 		business_ids = lv_reviews.filter(lv_reviews['user_id'] == friend).select('business_id')
# 		restaurants.append(business_ids.map(lambda b: b.business_id).collect())
# 	if len(restaurants) > 0:
# 		predictions[user.user_id] = random.choice(restaurants)

# def addPrediction(user_id): return prediction[user_id]
# predictionAdder = udf(addPrediction, StringType())

# business_graph = business_graph.edges.withColumn("predict_dst",predictionAdder(business_graph.edges['user_id']))
# new_edges = business_graph.select("src","predict_dst").withColumnRenamed("predict_dst","dst")
