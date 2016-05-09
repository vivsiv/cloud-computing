from pyspark.sql import SQLContext
from pyspark.sql.functions import lit

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
user = sqlContext.read.json("data/yelp_academic_dataset_user.json")
#user.count()
user_vertices = user.select("user_id").withColumnRenamed("user_id","id").withColumn("type",lit("user"))

#vertices data frame
vertices = user_vertices.unionAll(lv_vertices)

#load review data
review = sqlContext.read.json("data/yelp_academic_dataset_review.json")
lv_reviews = review.join(lv_restaurants.select("business_id"),"business_id").select("business_id","user_id","stars","date")

#edges data frame
edges = lv_reviews.withColumnRenamed("user_id","src").withColumnRenamed("business_id","dst")