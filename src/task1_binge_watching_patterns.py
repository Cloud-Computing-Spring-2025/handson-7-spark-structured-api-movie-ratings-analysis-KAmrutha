from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, round as spark_round
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType, BooleanType
import os

def initialize_spark(app_name="Task1_Binge_Watching_Patterns"):
    """Initialize and return a SparkSession."""
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    return spark

def load_data(spark, file_path):
    """Load the movie ratings data from a CSV file into a Spark DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please check the file path.")

    schema = StructType([
        StructField("UserID", IntegerType(), True),
        StructField("MovieID", IntegerType(), True),
        StructField("MovieTitle", StringType(), True),
        StructField("Genre", StringType(), True),
        StructField("Rating", FloatType(), True),
        StructField("ReviewCount", IntegerType(), True),
        StructField("WatchedYear", IntegerType(), True),
        StructField("UserLocation", StringType(), True),
        StructField("AgeGroup", StringType(), True),
        StructField("StreamingPlatform", StringType(), True),
        StructField("WatchTime", IntegerType(), True),
        StructField("IsBingeWatched", BooleanType(), True),
        StructField("SubscriptionStatus", StringType(), True)
    ])

    df = spark.read.csv(file_path, header=True, schema=schema)
    return df

def detect_binge_watching_patterns(df):
    """Identify the percentage of users in each age group who binge-watch movies."""
    binge_watchers = df.filter(col("IsBingeWatched") == True) \
        .groupBy("AgeGroup").agg(count("*").alias("BingeWatchers"))
    
    total_users = df.groupBy("AgeGroup").agg(count("*").alias("TotalUsers"))

    # Join the two DataFrames to calculate the percentage of binge watchers
    result_df = binge_watchers.join(total_users, "AgeGroup") \
        .withColumn("BingeWatchPercentage", spark_round((col("BingeWatchers") / col("TotalUsers")) * 100, 2))

    # Format the output to match the sample output (Age Group, Binge Watchers, and Percentage)
    result_df = result_df.select(
        "AgeGroup", 
        "BingeWatchers", 
        spark_round(col("BingeWatchPercentage"), 2).alias("Percentage")
    )

    return result_df

def write_output(result_df, output_path):
    """Write the result DataFrame to a CSV file."""
    result_df.coalesce(1).write.csv(output_path, header=True, mode='overwrite')

def main():
    """Main function to execute Task 1."""
    spark = initialize_spark()

    input_file = "/workspaces/handson-7-spark-structured-api-movie-ratings-analysis-KAmrutha/input/movie_ratings_data.csv"
    output_file = "/workspaces/handson-7-spark-structured-api-movie-ratings-analysis-KAmrutha/Outputs/binge_watching_patterns.csv"

    df = load_data(spark, input_file)
    result_df = detect_binge_watching_patterns(df)  
    write_output(result_df, output_file)

    spark.stop()

if __name__ == "__main__":
    main()
