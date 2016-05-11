# run this command with options to include GraphFrame
# $SPARK_HOME/bin/pyspark --packages graphframes:graphframes:0.1.0-spark1.6

from graphframes import *

vertices = sqlContext.createDataFrame([
  ("user", "18kPq7GPye-YQ3LyKyAZPw"),
  ("user", "rpOyqD_893cqmDAtJLbdog"),
  ("business", "zMN8UGd1zDEreT58OCdnyg"),
  ("business","UsFtqoBl7naz8AVUBZMjQQ"),], ["type","id"])

edges = sqlContext.createDataFrame([
  ("18kPq7GPye-YQ3LyKyAZPw", "zMN8UGd1zDEreT58OCdnyg", 5, "2012-08-01"),
], ["src", "dst", "stars", "date"])

# generate user-restaurant graph
g = GraphFrame(vertices, edges)
print g

g.vertices.show()
g.edges.show()