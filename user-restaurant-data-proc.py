from pyspark.sql import SQLContext
from pyspark.sql.functions import lit
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from graphframes import *
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
import random
import numpy as np
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, SQLTransformer

sqlContext = SQLContext(sc)

#load business data
business_df = sqlContext.read.json("data/yelp_academic_dataset_business.json")
#get just Vegas businesses
lv_business = business_df.filter(business_df["city"] == "Las Vegas")
# lv_business = business_df.filter(business_df["city"] == "Urbana")

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
trainNum = int(round(lv_reviews.count() * 0.6))
lv_reviews_train = sqlContext.createDataFrame(lv_reviews.orderBy("date").collect()[:trainNum])
lv_reviews_test = sqlContext.createDataFrame(lv_reviews.orderBy("date").collect()[trainNum:])

#user-business graph edges
edges = lv_reviews_train.withColumnRenamed("user_id","src").withColumnRenamed("business_id","dst")
test_edges = lv_reviews_test.withColumnRenamed("user_id","src").withColumnRenamed("business_id","dst").drop("stars").drop("date")

#### Cluster ####
num_of_clusters = 50
features_vec = lv_assembled.select("features_vec").map(lambda r: r.features_vec).collect()
clusters = KMeans.train(sc.parallelize(features_vec), num_of_clusters, maxIterations=10, runs=10,initializationMode="random")

# find highest rating restaurants for each user in testing set
g = GraphFrame(vertices, edges)
g_max = g.edges.select("src","dst","stars").groupBy("src").max("stars").withColumnRenamed("max(stars)","max_stars")
g_max_edge = g.edges.join(g_max,g.edges.src==g_max.src).select(g.edges.src,g.edges.dst,g.edges.stars,g_max.max_stars).filter("stars=max_stars").dropDuplicates(['src'])
g_max_edge_test = test_edges.join(g_max_edge,test_edges.src==g_max_edge.src).select(test_edges.src,g_max_edge.dst).dropDuplicates(['src'])

# add restaurant feature vectors to edge
g_features = g_max_edge_test.join(lv_assembled,g_max_edge_test.dst==lv_assembled.business_id).select(g_max_edge_test.src,g_max_edge_test.dst,lv_assembled.features_vec)

# append cluster_id to each restaurants in the edge
def addCluster(features_vec): return clusters.predict(features_vec)
clusterAdder = udf(addCluster,IntegerType())
g_all = g_features.withColumn("cluster_id", clusterAdder(g_features['features_vec']))
lv_all = lv_assembled.select("business_id","features_vec").withColumn("cluster_id", clusterAdder(g_features['features_vec']))

# construct cluster index
cluster_index = {}
for i in range(num_of_clusters):
	cluster_index[i] = lv_all.filter(lv_all.cluster_id == i).select("business_id","features_vec").map(lambda r: [r.business_id, r.features_vec]).collect()
	# cluster_index[i] = lv_all.filter(lv_all.cluster_id == i).select("business_id","features_vec").map(lambda r: r.business_id).collect()
	lv_all = lv_all.filter(lv_all.cluster_id != i)

def shareFeatures(f1,f2):
	f1 = f1.toArray()[2:]
	f2 = f2.toArray()[2:]
	if len([i for i,j in zip(f1,f2) if i == j and i != 0 and j != 0]) > 2: # have more than two common features
		return True
	else:
		return False

K = 3
# def addPrediction(cluster_id): return random.sample(cluster_index[cluster_id],K)
def addPrediction(features_vec,cluster_id):
	restaurants = cluster_index[cluster_id]
	possible_predictions = []
	for r in restaurants:
		if shareFeatures(r[1],features_vec):
			possible_predictions.append(r[0])
	if len(possible_predictions) <= K:
		return possible_predictions
	else:
		return random.sample(possible_predictions,K)

predictionAdder = udf(addPrediction, ArrayType(StringType()))
g_all = g_all.withColumn("predict_dst", predictionAdder(g_all['features_vec'],g_all['cluster_id']))
# g_all = g_all.withColumn("predict_dst", predictionAdder(g_all['cluster_id']))

# newly predicted edges
new_edges = g_all.select("src","predict_dst").withColumnRenamed("predict_dst","dst")

# evaluation
# def apk(actual, predicted):
#     score = 0.0
#     num_hits = 0.0
#     for i,p in enumerate(predicted):
#         if p in actual and p not in predicted[:i]:
#             num_hits += 1.0
#             score += num_hits / (i+1.0)
#     if not actual:
#         return 0.0
#     return score / min(len(actual),len(predicted))
# def mapk(actual, predicted):
#     return np.mean([apk(a,p) for a,p in zip(actual, predicted)])

test_edge_index = {}
for e in test_edges.collect():
	if e.src in test_edge_index:
		test_edge_index[e.src].append(e.dst)
	else:
		test_edge_index[e.src] = [e.dst]

correct_prediction = 0
relevant_links_count = len(test_edge_index)
predicted_links_count = new_edges.count() * K
# actual_list = []
# predicted_list = []
for e in new_edges.collect():
	if e.src in test_edge_index:
		if len(e.dst) != 0:
			# actual_list.append(test_edge_index[e.src])
			# predicted_list.append(e.dst)
			for dst in e.dst:
				if dst in test_edge_index[e.src]:
					correct_prediction += 1
					continue
precision = float(correct_prediction) / predicted_links_count # 85 / 17690 = 0.0048
recall = float(correct_prediction) / relevant_links_count

F1 = 2 * precision * recall / (precision + recall)

print precision # 50,3: 0.002336536649707933; 50,3,filter features:0.00265686828717; 50,3,filter features[2:]=0.010194; 200: 0.00508762012436
print recall
print F1
# print mapk(actual_list, predicted_list) # 50,3: 0.002214; 50,3,filter features:0.002206; 50,3,filter features[2:]=0.003409; 200: 0.00541825157505
#Urbana (0.0263157894737, 0.0681481481481)
'''
Result:
Urbana (precision,recall,F1)
1. 50 clusters, K=3, shareFeatures: (0.0350877,0.0095,0.01495327)
2. 100 clusters, K=3, shareFeatures: (0.0263, 0.00713, 0.011215)
3. 200 clusters, K=3, shareFeatures: same as 100 clusters

LV (precision,recall,F1)
1. 50 clusters, K=1: (0.0027699, 0.000524,0.000881)
2. 50 clusters, K=3: (0.0026757,0.001518684,0.0019376)
3. 50 clusters, K=3, shareFeatures: (0.003222,0.0018288,0.002333)
4. 100 clusters, K=3, shareFeatures: (0.0041643,0.00236358,0.00301558)
5. 200 clusters, K=3, shareFeatures: (0.0054,0.003069,0.003916)

'''
