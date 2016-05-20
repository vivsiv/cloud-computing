from pyspark.sql import SQLContext
from pyspark.sql.functions import lit
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from graphframes import *
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
import random
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, SQLTransformer

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

#initial feature set
# lv_clustering_data = lv_restaurants.map(lambda r:
# 		[r.business_id, "None" if len(r.neighborhoods) is 0 else r.neighborhoods[0], r.stars, r.attributes['Price Range']]
# 	).toDF(['business_id','neighborhood','stars','price_range'])


#expanded feature set
clustering_columns = ['business_id',
			'neighborhood',
			'stars',
			'price_range',
			'dinner',
			'lunch',
			'breakfast',
			'romantic',
			'upscale',
			'casual',
			'alcohol',
			'take_out']

lv_clustering_data = lv_restaurants.map(lambda r:
		[r.business_id,
		 "None" if len(r.neighborhoods) is 0 else r.neighborhoods[0], 
		 r.stars, r.attributes['Price Range'], 
		 False if r.attributes['Good For'] is None else r.attributes['Good For']['dinner'],
		 False if r.attributes['Good For'] is None else r.attributes['Good For']['lunch'],
		 False if r.attributes['Good For'] is None else r.attributes['Good For']['breakfast'],
		 False if r.attributes['Ambience'] is None else r.attributes['Ambience']['romantic'],
		 False if r.attributes['Ambience'] is None else r.attributes['Ambience']['upscale'],
		 False if r.attributes['Ambience'] is None else r.attributes['Ambience']['casual'],
		 False if (r.attributes['Alcohol'] is None or r.attributes['Alcohol'] == 'none') else True,
		 False if r.attributes['Take-out'] is None else r.attributes['Take-out']]
	).toDF(clustering_columns)

# drop row with null values
lv_clustering_data = lv_clustering_data.dropna()

#Neighborhood feature engineering
stringIndexer = StringIndexer(inputCol="neighborhood", outputCol="neigh_index")
lv_model = stringIndexer.fit(lv_clustering_data)
lv_indexed = lv_model.transform(lv_clustering_data)
encoder = OneHotEncoder(dropLast=False, inputCol="neigh_index", outputCol="neigh_vec")
lv_encoded = encoder.transform(lv_indexed)

#initial feature set
# assembler = VectorAssembler(
#     inputCols=["stars", "price_range", "neigh_vec"],
#     outputCol="features_vec")

#expanded feature set
feature_columns = clustering_columns[2:]
feature_columns.append("neigh_vec")
assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features_vec")

lv_assembled = assembler.transform(lv_encoded)


# neighborhoods = lv_clustering_data.groupBy("neighborhood").count().select("neighborhood").collect()
# for n in neighborhoods:
# 	statement = "SELECT *, neighborhood='{0}' AS {0} FROM __THIS__".format(n['neighborhood'].replace(" ","_"))
# 	sqlTrans = SQLTransformer(statement=statement)
# 	lv_clustering_data = sqlTrans.transform(lv_clustering_data)
#
# col_drop_list = ['business_id', 'neighborhood']
# lv_clustering = lv_clustering_data.select([column for column in lv_clustering_data.columns if column not in col_drop_list])


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
num_of_clusters = 50
features_vec = lv_assembled.select("features_vec").map(lambda r: r.features_vec).collect()
clusters = KMeans.train(sc.parallelize(features_vec), num_of_clusters, maxIterations=10, runs=10,initializationMode="random")

# find highest rating restaurants for each user
g = GraphFrame(vertices, edges)
g_max = g.edges.select("src","dst","stars").groupBy("src").max("stars").withColumnRenamed("max(stars)","max_stars")
g_max_edge = g.edges.join(g_max,g.edges.src==g_max.src).select(g.edges.src,g.edges.dst,g.edges.stars,g_max.max_stars).filter("stars=max_stars").dropDuplicates(['src'])

# add restaurant feature vectors to edge
g_features = g_max_edge.join(lv_assembled,g_max_edge.dst==lv_assembled.business_id).select(g_max_edge.src,g_max_edge.dst,lv_assembled.features_vec)

# append cluster_id to each restaurants in the edge
def addCluster(features_vec): return clusters.predict(features_vec)
clusterAdder = udf(addCluster,IntegerType())
g_all = g_features.withColumn("cluster_id", clusterAdder(g_features['features_vec']))
lv_all = lv_assembled.select("business_id","features_vec").withColumn("cluster_id", clusterAdder(g_features['features_vec']))

# construct cluster index
cluster_index = {}
for i in range(num_of_clusters):
	cluster_index[i] = lv_all.filter(lv_all.cluster_id == i).select("business_id").map(lambda r: r.business_id).collect()
	lv_all = lv_all.filter(lv_all.cluster_id != i)

# list of all edges
all_edges = g.edges.select("src","dst").map(lambda r: array([r.src,r.dst])).collect()

def addPrediction(cluster_id): return random.choice(cluster_index[cluster_id])
# def addPrediction(user_id,cluster_id):
	# predict_dst = random.choice(cluster_index[cluster_id])
	# while filter(lambda x: x[0]==user_id and x[1]==predict_dst,all_edges) != []:
	# 	predict_dst = random.choice(cluster_index[cluster_id])
	# return predict_dst
predictionAdder = udf(addPrediction, StringType())
# g_all = g_all.withColumn("predict_dst", predictionAdder(g_all['src'],g_all['cluster_id']))
g_all = g_all.withColumn("predict_dst", predictionAdder(g_all['cluster_id']))

# newly predicted edges
new_edges = g_all.select("src","predict_dst").withColumnRenamed("predict_dst","dst")

# todo
# filter out the restaurants the user has already visited
