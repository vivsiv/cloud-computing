from pyspark.sql import SQLContext

sqlContext = SQLContext(sc)

#load business data
business_df = sqlContext.read.json("data/yelp_academic_dataset_business.json")


#print out the business table's schema
business_df.printSchema()

#get just Vegas businesses
lv_business = business_df.filter(business_df["city"] == "Las Vegas")

#get just vegas restaurants
lv_restaurants = lv_business.select("*").where("array_contains(categories,'Restaurants')")
lv_restaurants.count()
lv_vertices = lv_restaurants.select("business_id").withColumnRenamed("business_id","id").withColumn("type",lit("business"))

#4658 #Add more categories?


#load checkin data
checkin = sqlContext.read.json("data/yelp_academic_dataset_checkin.json")

#join las vegas restaurants to checkins
restaurant_checkin = lv_restaurants.join(checkin_df,"business_id")

#load review data
review = sqlContext.read.json("data/yelp_academic_dataset_review.json")

#join reviews to restaurants
restaurant_review = lv_restaurants.join(review_df,"business_id")

#load user data
user = sqlContext.read.json("data/yelp_academic_dataset_user.json")
user.count()
user_vertices = user.select("user_id").withColumnRenamed("user_id","id").withColumn("type",lit("user"))

vertices = user_vertices.unionAll(lv_vertices)

#join users to checkins
user_checkin_df = user_df.join(checkin_df,"user_id")

#join users to reviews
user_checkin_df = user_df.join(checkin_df,"user_id")