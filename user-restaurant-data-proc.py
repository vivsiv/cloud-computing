from pyspark.sql import SQLContext
from pyspark.sql.functions import lit
from pyspark.mllib.clustering import KMeans, KMeansModel

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
neighborhoods = lv_clustering_data.groupBy("neighborhood").count().select("neighborhood")
# def n_map(data,n):
# 	data = data.withColumn(n,False)

# neighborhoods.map(n_map(lv_clustering_data))

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



