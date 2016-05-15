from pyspark.sql import SQLContext
from pyspark.sql.functions import lit
from pyspark.sql.functions import udf
from graphframes import *
from pyspark.mllib.clustering import KMeans, KMeansModel
# from pyspark.ml.feature import VectorIndexer
from pyspark.ml.feature import OneHotEncoder, StringIndexer

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
stringIndexer = StringIndexer(inputCol="neighborhood", outputCol="neighborhoodIndex")
lv_model = stringIndexer.fit(lv_clustering_data)
lv_indexed = lv_model.transform(lv_clustering_data)
encoder = OneHotEncoder(dropLast=False, inputCol="neighborhoodIndex", outputCol="neighborhoodVec")
lv_encoded = encoder.transform(lv_indexed)


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
