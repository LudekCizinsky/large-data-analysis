## About
In this exercise, the goal was to get familiar working with Apache spark.

## Exercise 1

Log into the ambari cluster. Find out the size of /datasets/retail/retail.csv in hdfs (hint: file system shell reference):

(a) What command do you use? What are the size(s)?

(b) Can you change directory to /datasets in hdfs? Explain why.

**Solution** (checked)

---

According to the [documentation](https://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-common/FileSystemShell.html), we can use the following command:

```
hadoop fs -du -h "/datasets/retail/retail.csv"
```

As a result, we obtain the following output:

```
49.3 M  148.0 M  /datasets/retail/retail.csv
```

This output follows the following pattern:

```
size disk_space_consumed_with_all_replicas full_path_name
```

Note that the option `-h` means that we want the output in a human readable format.

To answer the second question, we simply **can not**. This is because we are at
a name node which only consists of meta data about the data we store. Therefore,
the actual data is actually physically stored on the other nodes within the
`amber` cluster.

## Exercise 2

Using `pyspark`, answer the following questions:

(a) What is the number of rows in retail.csv?

(b) What is the schema of the data in this file?

(c) Find the items that are frequently bought together

**Solution** (checked)

---

The code to answer the questions is specified below with corresponding comments.
Note that for the last question, I followed the following [tutorial](https://towardsdatascience.com/market-basket-analysis-using-pysparks-fpgrowth-55c37ebd95c0).

```python
import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import functions as F

# Set up for the spark session
conf = SparkConf()
conf.set("spark.executor.memory", "2G")
conf.set("spark.executor.instances", "2")
conf.set("spark.app.name", "ludekcizinsky-ex5")
conf.set("spark.ui.enabled", False)

# Build the session 
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# Read the csv-file from HDFS
df = spark.read.option("header",True).csv("/datasets/retail/retail.csv")

# Get the number of rows 
print(f"> Number of rows in the dataset is: {df.count()}")

# Show the schema of the data
print(f"> Schema of the dataset is:")
df.printSchema()

# Find the items that are frequently bought together
print("> Frequent items bought together: ")
## Preprocessing neccesarry for the algorithm to run
df = df.dropDuplicates(['InvoiceNo', 'StockCode']).sort('InvoiceNo')
## Start with groupping data and aggegating, the result should be as follows:
## InvoiceNo: [StockCode1, ..., StockCodeN]
dfg = df.groupBy("InvoiceNo").agg(F.collect_list("StockCode")).sort('InvoiceNo')

## Train the model
fpGrowth = FPGrowth(itemsCol="collect_list(StockCode)", minSupport=0.006, minConfidence=0.006)
model = fpGrowth.fit(dfg)

# Display frequent itemsets
model.freqItemsets.show()
```

Note that there is a more elegant way in a sense that instead of writing `collect_list`, you can write `collect_set`.
Therefore you no longer need to do the preprocessing part. Finally, the output of this code is:

```
> Number of rows in the dataset is: 541909
> Schema of the dataset is:
root
 |-- _c0: string (nullable = true)
 |-- InvoiceNo: string (nullable = true)
 |-- StockCode: string (nullable = true)
 |-- Description: string (nullable = true)
 |-- Quantity: string (nullable = true)
 |-- InvoiceDate: string (nullable = true)
 |-- UnitPrice: string (nullable = true)
 |-- CustomerID: string (nullable = true)
 |-- Country: string (nullable = true)

> Frequent items bought together:

+--------------------+----+
|               items|freq|
+--------------------+----+
|             [22633]| 487|
|      [22633, 22866]| 215|
|      [22633, 22865]| 238|
|      [22633, 22867]| 186|
|             [23236]| 344|
|      [23236, 23240]| 160|
|             [23158]| 258|
|             [21922]| 209|
|             [21471]| 166|
|            [85123A]|2246|
|             [23504]| 208|
|             [23157]| 166|
|             [22423]|2172|
|     [22423, 85123A]| 355|
|             [22667]| 486|
|      [22667, 22666]| 218|
|             [22579]| 343|
|      [22579, 22578]| 282|
|[22579, 22578, 22...| 232|
|      [22579, 22577]| 250|
+--------------------+----+
only showing top 20 rows
```

