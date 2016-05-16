from pyspark.sql import SQLContext
from pyspark.sql.functions import lit
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from graphframes import *
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
import random
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

sqlContext = SQLContext(sc)

#load business data
business_df = sqlContext.read.json("data/yelp_academic_dataset_business.json")
#get just Vegas businesses
lv_business = business_df.filter(business_df["city"] == "Las Vegas")
#get just vegas restaurants
lv_restaurants = lv_business.select("*").where("array_contains(categories,'Restaurants')")
#lv_restaurants.count()
#4658 #Add more categories?

#get vegas clustering data
lv_clustering_data = lv_restaurants.map(lambda r:
		[r.business_id,"N/A" if len(r.neighborhoods) is 0 else r.neighborhoods[0],r.stars,r.attributes['Price Range']]
	).toDF(['business_id','neighborhood','stars','price_range'])

#Neighborhood feature engineering
stringIndexer = StringIndexer(inputCol="neighborhood", outputCol="neigh_index")
lv_model = stringIndexer.fit(lv_clustering_data)
lv_indexed = lv_model.transform(lv_clustering_data)
encoder = OneHotEncoder(dropLast=False, inputCol="neigh_index", outputCol="neigh_vec")
lv_encoded = encoder.transform(lv_indexed)

assembler = VectorAssembler(
    inputCols=["stars", "price_range", "neigh_vec"],
    outputCol="features_vec")
lv_assembled = assembler.transform(lv_encoded)


# neighborhoods = lv_clustering_data.groupBy("neighborhood").count().select("neighborhood").collect()

# for n in neighborhoods:
# 	lv_clustering_data = lv_clustering_data.withColumn(n['neighborhood'],lit(False))



#generate lv_restaurant vertices set
lv_vertices = lv_restaurants.select("business_id").withColumnRenamed("business_id","id").withColumn("type",lit("business"))

#load user data
users = sqlContext.read.json("data/yelp_academic_dataset_user.json")
#user.count()
#552339
#generate user vertices
users_vertices = users.select("user_id").withColumnRenamed("user_id","id").withColumn("type",lit("user"))

#user-business graph vertices
vertices = users_vertices.unionAll(lv_vertices)

#load review data
review = sqlContext.read.json("data/yelp_academic_dataset_review.json")
lv_reviews = review.join(lv_restaurants.select("business_id"),"business_id").select("business_id","user_id","stars","date")

#user-business graph edges
edges = lv_reviews.withColumnRenamed("user_id","src").withColumnRenamed("business_id","dst")

#### Cluster ####
# create cluter for testing use
num_of_clusters = 10
test = lv_clustering_data.select("stars","price_range")
clusters = KMeans.train(test.dropna().map(list), num_of_clusters, maxIterations=10, runs=10,initializationMode="random")

# find highest rating restaurants for each user
g = GraphFrame(vertices, edges)
gmax = g.edges.select("src","dst","stars").groupBy("src").max("stars").withColumnRenamed("max(stars)","max_stars")
gjoin = g.edges.join(gmax,g.edges.src==gmax.src).select(g.edges.src,g.edges.dst,g.edges.stars,gmax.max_stars).filter("stars=max_stars").dropDuplicates(['src'])

# add restaurant feature vectors to edge
gjoin_restaurant_info = gjoin.join(lv_clustering_data,gjoin.dst==lv_clustering_data.business_id).select(gjoin.src,gjoin.dst,gjoin.max_stars,lv_clustering_data.neighborhood,lv_clustering_data.price_range,lv_clustering_data.stars)
def addCluster(stars,price_range): return clusters.predict(array([stars,price_range]))
clusterAdder = udf(addCluster,IntegerType())
g_all = gjoin_restaurant_info.withColumn("cluster_id", clusterAdder(gjoin_restaurant_info['stars'],gjoin_restaurant_info['price_range']))
lv_all = lv_clustering_data.withColumn("cluster_id", clusterAdder(lv_clustering_data['stars'],lv_clustering_data['price_range']))

###### runnable ########
# construct cluster index
cluster_index = {}
for i in range(num_of_clusters):
	cluster_index[i] = lv_all.filter(lv_all.cluster_id == i).select("business_id").map(lambda r: r.business_id).collect()

def addPrediction(cluster_id): return random.choice(cluster_index[cluster_id])
predictionAdder = udf(addPrediction, StringType())
g_all = g_all.withColumn("predict_dst", predictionAdder(g_all['cluster_id']))

new_edges = g_all.select("src","predict_dst").withColumnRenamed("predict_dst","dst")

# get list of all restaurants
# candidates = lv_reviews.select("business_id").dropDuplicates().map(lambda r: r.business_id).collect() #4658
#
# # add a candidate column to gjoin_restaurant_info
# def addCandidate(arg): return candidates
# candidateAdder = udf(addCandidate,ArrayType(StringType()))
# g_edge_all = gjoin_restaurant_info.withColumn("candidates", candidateAdder(gjoin_restaurant_info['src']))
#
# # find restaurants in the same clusters
# def findSameCluster(row):
#     fv1 = array([row.stars,row.price_range])
#     def f(c):
#         r = lv_clustering_data.filter(lv_clustering_data.business_id==c).select("stars","price_range").first()
#         fv2 = array([r.stars,r.price_range])
#         return clusters.predict(fv1) == clusters.predict(fv2)
#     filter(f,row.candidates)
#
# g_edge_all.foreach(findSameCluster)


# 2. add the edges for these user-restaurant pairs
