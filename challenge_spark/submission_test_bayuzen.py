# ---- Import Library -----
import pyspark
import os
import json
import argparse

from dotenv import load_dotenv
from pathlib import Path
from pyspark.sql.types import StructType,IntegerType,StringType
from pyspark.sql import functions as f
from pyspark.sql.functions import (col, 
                                   udf, 
                                   pandas_udf,
                                   to_timestamp,
                                   col,
                                   when,
                                   to_date,
                                   year, 
                                   month,
                                   dayofmonth, 
                                   hour, 
                                   quarter, 
                                   date_format,
                                   dayofweek,
                                   datediff,
                                   trunc,
                                   asc,
                                  desc,
                                  corr,
                                   round,
                                  min,
                                   sum,
                                   max,
                                  countDistinct,
                                   collect_set
                                  )
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Settip Path -----
dotenv_path = Path('/resources/.env')
load_dotenv(dotenv_path=dotenv_path)
postgres_host = os.getenv('DIBIMBING_DE_POSTGRES_HOST')
postgres_db = os.getenv('DIBIMBING_DE_POSTGRES_DB')
postgres_user = os.getenv('DIBIMBING_DE_POSTGRES_ACCOUNT')
postgres_password = os.getenv('DIBIMBING_DE_POSTGRES_PASSWORD')


# ---- Setup Spark -----
sparkcontext = pyspark.SparkContext.getOrCreate(conf=(
        pyspark
        .SparkConf()
        .setAppName('Dibimbing')
        .setMaster('local')
        .set("spark.jars", "/opt/postgresql-42.2.18.jar")
    ))
sparkcontext.setLogLevel("WARN")

spark = pyspark.sql.SparkSession(sparkcontext.getOrCreate())

# ---- Load Dataset -----
jdbc_url = f'jdbc:postgresql://{postgres_host}/{postgres_db}'
jdbc_properties = {
    'user': postgres_user,
    'password': postgres_password,
    'driver': 'org.postgresql.Driver',
    'stringtype': 'unspecified'
}

retail_df = spark.read.jdbc(
    jdbc_url,
    'public.retail',
    properties=jdbc_properties
)

retail_df.createOrReplaceTempView("retail_data")


# ----- Main processs -----
#Show Data
spark.sql("""
 select * from retail_data
""").show()

# --- Data Cleaning ----
# Apakah terdapat Revenue yang negatif ?
spark.sql('''
          select min(quantity * unitprice) as revenue
          from retail_data
          having revenue < 0
          ''').show()
df = spark.sql('''
    select 
        invoiceno,
        stockcode,
        invoicedate,
        customerid,
        description,
        quantity,
        unitprice,
        round(quantity * unitprice) as TotalSales,
        country
    from
        retail_data
    where 
        invoiceno not like("%C%")
        and customerid is not null 
        and (quantity * unitprice) > 0
        and invoicedate < '2011-12-01'
        and stockcode not in ("BANK CHARGES","DOT","C2","M","POST","PADS")
''')

# --- EDA ----
#summary statistics
df\
    .select(col('quantity'),
            col('unitprice'),
            col('TotalSales'))\
    .describe().show()

# Feature Engineering
# Convert InvoiceDate column to timestamp
df = df.withColumn("InvoiceDate", df["InvoiceDate"].cast("timestamp"))

# Extract year, month, day, hour, quarter, and formatted date columns
df = df.withColumn("year", year(df["InvoiceDate"]))
df = df.withColumn("months", month("InvoiceDate"))
df = df.withColumn("days", dayofmonth("InvoiceDate"))
df = df.withColumn("weekdays", dayofweek("InvoiceDate"))
df = df.withColumn("quarter", quarter("InvoiceDate"))
df = df.withColumn("InvoiceYearMonth", date_format("InvoiceDate", "MMM-yyyy"))

## Analysis 1 : What country which have high purchases ?
# Group by Country and calculate total quantity
country_order = df.groupBy("country").count().withColumnRenamed("count", "OrderCount")
country_order = country_order.orderBy(f.desc("OrderCount")).limit(10)

# Group by Country and calculate total quantity
country_quantity = df.groupBy("country").sum("quantity").withColumnRenamed("sum(quantity)", "TotalQuantity")
country_quantity = country_quantity.orderBy(f.desc("TotalQuantity")).limit(10)

# Group by Country and calculate total sales
country_profit = df.groupBy("country").sum("TotalSales").withColumnRenamed("sum(TotalSales)", "TotalSales")
country_profit = country_profit.orderBy(f.desc("TotalSales")).limit(10)

# Convert Spark DataFrame to Pandas DataFrame for visualization
country_order_pd = country_order.toPandas()
country_quantity_pd = country_quantity.toPandas()
country_profit_pd = country_profit.toPandas()

# Plotting
plt.figure(figsize=(10, 15), dpi=100)
plt.suptitle("Top 10 Country Most Order, Quantity, Profit", fontsize=15, y=1)

plt.subplot(311)
sns.barplot(data=country_order_pd, x="country", y="OrderCount", palette="Set2", edgecolor='k')
plt.title("Top 10 Country Most Ordered", fontsize=12)
plt.xticks(rotation=45)
plt.ylabel("Count")

plt.subplot(312)
sns.barplot(data=country_quantity_pd, x="country", y="TotalQuantity", palette="Set2", edgecolor='k')
plt.title("Top 10 Country Most Quantity", fontsize=12)
plt.xticks(rotation=45)

plt.subplot(313)
sns.barplot(data=country_profit_pd, x="country", y="TotalSales", palette="Set2", edgecolor='k')
plt.title("Top 10 Country Most Profitable", fontsize=12)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


## filter only contain United Kingdom
df_uk = df.filter(col("country") == "United Kingdom")
df_uk.createOrReplaceTempView('table_uk')

## Analysis 2 : Quantity and Total Purchases Grotwh By year
# Calculate total sales and total quantity per year using Spark SQL
qt_pd = spark.sql("""
    SELECT year, SUM(quantity) AS TotalQuantity, SUM(TotalSales) AS TotalSales
    FROM table_uk
    GROUP BY year
    ORDER BY year
""").toPandas()


# Plotting
fig = plt.figure(figsize=(8, 5), dpi=100)
ax1 = plt.subplot(111)
ax1.bar(qt_pd["year"], qt_pd["TotalSales"], color="blue", label="Total Sales")
plt.xticks(qt_pd["year"])
ax2 = ax1.twinx()
ax2.plot(qt_pd["year"], qt_pd["TotalQuantity"], marker="o", color="r", label="Total Quantity")

plt.title("Revenue and Quantity in 2010 vs 2011")
ax1.legend(loc="best", bbox_to_anchor=(1.05, 0.5, 0., 0.5))
ax2.legend(loc="best", bbox_to_anchor=(1.3, 0.5, 0., 0.43))
plt.show()

# filter data
df_uk = df_uk.filter(col('year') > 2010)

# Calculate total sales by month
total_sales_month = df_uk.groupBy("months") \
    .sum("TotalSales") \
    .orderBy("months") \
    .select("months" , "sum(TotalSales)")


# Calculate total sales by quarter
total_sales_quarter = df_uk.groupBy("quarter") \
    .sum("TotalSales") \
    .orderBy("quarter")

# Calculate total sales by day
total_sales_day = df_uk.groupBy("days") \
    .sum("TotalSales") \
    .orderBy("days")

# Calculate total sales by day of week
total_sales_weekday = df_uk.groupBy("weekdays") \
    .sum("TotalSales") \
    .orderBy("weekdays")\
    .withColumn('weekdays',when(col("weekdays") == 1, "Sunday")
                            .when(col("weekdays") == 2, "Monday")
                            .when(col("weekdays") == 3, "Tuesday")
                            .when(col("weekdays") == 4, "Wednesday")
                            .when(col("weekdays") == 5, "Thursday")
                            .when(col("weekdays") == 6, "Friday")
                            .when(col("weekdays") == 7, "Saturday"))

# Convert Spark DataFrames to Pandas DataFrames
total_sales_month_pd = total_sales_month.toPandas()
total_sales_quarter_pd = total_sales_quarter.toPandas()
total_sales_day_pd = total_sales_day.toPandas()
total_sales_weekday_pd = total_sales_weekday.toPandas()


# Plotting the data
plt.figure(figsize=(16, 12))

plt.subplot(3, 2, 1)
plt.plot(total_sales_month_pd['months'], total_sales_month_pd['sum(TotalSales)'], marker='o', color='lightseagreen')
plt.axvline(11, color='k', linestyle='--', alpha=0.3)
plt.text(8, 0.97e6, "Most Total Sales")
plt.title("Total Sales by Month")

plt.subplot(3, 2, 2)
plt.bar(total_sales_quarter_pd['quarter'], total_sales_quarter_pd['sum(TotalSales)'], color='darkslategrey')
plt.title("Total Sales by Quarter")

plt.subplot(3, 2, 3)
plt.plot(total_sales_day_pd['days'], total_sales_day_pd['sum(TotalSales)'], marker='o')
plt.axvline(x=20, linestyle="--", color='blue')
plt.title("Total Sales by Day")

plt.subplot(3, 2, 4)
plt.bar(total_sales_weekday_pd['weekdays'], total_sales_weekday_pd['sum(TotalSales)'], color='darkorange')
plt.title("Total Sales by Day of Week")

plt.tight_layout()
plt.show()

# Filter data for November
df_november = df_uk.filter(col("months") == 11)

# Group by description and calculate total quantity sold
product_sales_november = df_november.groupBy("Description").agg({"Quantity": "sum", "TotalSales": "sum","unitprice":"mean"}) \
    .withColumnRenamed("sum(Quantity)", "TotalQuantity") \
    .withColumnRenamed("sum(TotalSales)", "TotalSales") \
    .withColumnRenamed("avg(unitprice)","MeanPrice")\
    .orderBy(desc("TotalQuantity"), desc("TotalSales"),desc("MeanPrice"))


# Show the top products sold in November
product_sales_november.show()


## Analysis 4 : Statistics Correlation Quantity and Unit Price
# Calculate Pearson's correlation coefficient between Quantity and UnitPrice
correlation = df_uk.select(corr("Quantity", "UnitPrice")).collect()[0][0]

print("The Pearson's Correlation Coefficient is:", '{:.2f}'.format(correlation))

## Analysis 5 : Retention And Frequency Purchase
# Convert InvoiceDate column to date type
df_uk = df_uk.withColumn("InvoiceDate", to_date(col("InvoiceDate")))

# Group by InvoiceNo and InvoiceDate, and aggregate TotalSales and CustomerID
invoice_customer = df_uk.groupBy("InvoiceNo", "InvoiceDate").agg(
    sum("TotalSales").alias("TotalSales"),
    max("customerid").alias("CustomerID")
)

# Calculate the count of repeat customers per month
window_spec = Window.partitionBy(year("InvoiceDate").alias("Year"), month("InvoiceDate").alias("Month")).orderBy("CustomerID")
monthly_repeat = invoice_customer.groupBy(year("InvoiceDate").alias("Year"), month("InvoiceDate").alias("Month"), "CustomerID") \
    .agg(countDistinct("InvoiceNo").alias("InvoiceCount")) \
    .filter(col("InvoiceCount") > 1) \
    .groupBy("Year", "Month") \
    .agg(countDistinct("CustomerID").alias("RepeatCustomers"))

# Calculate the count of unique customers per month
monthly_unique_customers = df_uk.groupBy(year("InvoiceDate").alias("Year"), month("InvoiceDate").alias("Month")) \
    .agg(countDistinct("CustomerID").alias("AllCustomers"))

# Calculate the percentage of repeat customers
monthly_repeat_percentage = monthly_repeat.join(monthly_unique_customers, ["Year", "Month"], "inner") \
    .withColumn("RepeatPercentage", (col("RepeatCustomers") / col("AllCustomers")) * 100)

# Convert Spark DataFrame to Pandas DataFrame for plotting
monthly_repeat_pd = monthly_repeat.select("Year", "Month", "RepeatCustomers").toPandas()
monthly_unique_customers_pd = monthly_unique_customers.select("Year", "Month", "AllCustomers").toPandas().sort_values("Month",ascending=True)
monthly_repeat_percentage_pd = monthly_repeat_percentage.select("Year", "Month", "RepeatPercentage").toPandas().sort_values("Month",ascending=True)

# Plot the number of customers over time
fig, ax1 = plt.subplots(figsize=(10, 7))
ax2 = ax1.twinx()

monthly_unique_customers_pd.plot(x="Month", y="AllCustomers", ax=ax1, legend=False, grid=True)
monthly_repeat_percentage_pd.plot(x="Month", y="RepeatPercentage", kind="bar", ax=ax2, legend=False, color="green", alpha=0.2)

ax1.set_xlabel("Month")
ax1.set_ylabel("Number of Customers")
ax2.set_ylabel("Percentage of Repeat Customers")

ax1.legend(["All Customers"])
ax2.legend(["Percentage of Repeat"], loc="upper right")

plt.title("Number of All Customers and Percentage of Repeat Customers Over Time")
plt.xticks(rotation=45)

plt.show()

