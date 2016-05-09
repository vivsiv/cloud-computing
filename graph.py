# $SPARK_HOME/bin/pyspark --packages graphframes:graphframes:0.1.0-spark1.6

from graphframes import *

# input would be two json files, one for vertices, another for edges
# format for vertices.json
# {"id":"18kPq7GPye-YQ3LyKyAZPw", "type":"user"}
# format for edges.json
# {"src":18kPq7GPye-YQ3LyKyAZPw,"dst":"zMN8UGd1zDEreT58OCdnyg","stars":5,"date":"2012-08-01"}

# creates a DataFrame based on the content of a JSON file
# vertices = sqlContext.read.json("data/vertices.json")
# vertices.show()
# vertices.printSchema()

# edges = sqlContext.read.json("data/edges.json")
# edges.show()
# edges.printSchema()

vertices = sqlContext.createDataFrame([
  ("user", "18kPq7GPye-YQ3LyKyAZPw"),
  ("user", "rpOyqD_893cqmDAtJLbdog"),
  ("business", "zMN8UGd1zDEreT58OCdnyg"),
  ("business","UsFtqoBl7naz8AVUBZMjQQ"),], ["type","id"])

edges = sqlContext.createDataFrame([
  ("18kPq7GPye-YQ3LyKyAZPw", "zMN8UGd1zDEreT58OCdnyg", 5, "2012-08-01"),
], ["src", "dst", "stars", "date"])

g = GraphFrame(vertices, edges)
print g

g.vertices.show()
g.edges.show()