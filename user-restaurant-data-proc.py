from pyspark.sql import SQLContext
from pyspark.sql.functions import lit
from pyspark.sql.functions import udf
from graphframes import *

sqlContext = SQLContext(sc)

#load business data
business_df = sqlContext.read.json("data/yelp_academic_dataset_business.json")


#print out the business table's schema
#business_df.printSchema()

#get just Vegas businesses
lv_business = business_df.filter(business_df["city"] == "Las Vegas")

#get just vegas restaurants
lv_restaurants = lv_business.select("*").where("array_contains(categories,'Restaurants')")
#lv_restaurants.count()
#4658 #Add more categories?

lv_vertices = lv_restaurants.select("business_id").withColumnRenamed("business_id","id").withColumn("type",lit("business"))

#load user data
users = sqlContext.read.json("data/yelp_academic_dataset_user.json")
#user.count()
users_vertices = users.select("user_id").withColumnRenamed("user_id","id").withColumn("type",lit("user"))

#user-business graph vertices
vertices = users_vertices.unionAll(lv_vertices)

#load review data
review = sqlContext.read.json("data/yelp_academic_dataset_review.json")
lv_reviews = review.join(lv_restaurants.select("business_id"),"business_id").select("business_id","user_id","stars","date")

#user-business graph edges
edges = lv_reviews.withColumnRenamed("user_id","src").withColumnRenamed("business_id","dst")

#### Cluster ####
# find highest rating restaurants for each user
g = GraphFrame(vertices, edges)
gmax = g.edges.select("src","dst","stars").groupBy("src").max("stars")
gmax = gmax.withColumnRenamed("max(stars)","max_stars")
gjoin = g.edges.join(gmax,g.edges.src==gmax.src).select(g.edges.src,g.edges.dst,g.edges.stars,gmax.max_stars)
gjoin = gjoin.filter("stars=max_stars")

# find restaurants in the same clusters
# 1. add to gjoin a column = list of restaurants in the same clusters
def findSameCluster(rid):
    ## code for finding restaurants in same clusters ##

    ###
    a = [1,2,3]
    return a
clusterFinder = udf(findSameCluster)
lv_reviews.withColumn("candidate",clusterFinder(lv_reviews['business_id']))
# 2. add the edges for these user-restaurant pairs
