import boto3
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql import DataFrame

def init_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)


class SparkData:
    def __init__(self, data):
        self.sdf = data
        self.unique_users = None
        self.unique_users_count = None

    def clean_sdf(self):
        self.sdf = self.sdf.dropna(subset="userId")
        self.unique_users = self.sdf.select("userId").distinct()
        self.unique_users_count = self.unique_users.count()
        return self
    
    def build_song_counts(self):
        self.song_counts = (
            self.sdf[["userId", "artist"]].dropna(subset="artist").groupBy("userId").count()
        )
        self.song_counts = self.song_counts.withColumnRenamed("count", "song_counts")

        self.song_counts = self.handle_missing_users(self.song_counts)
        
    def handle_missing_users(self, sdf: DataFrame) -> DataFrame:
        """
        Handle missing users in a PySpark DataFrame.

        Parameters:
        - sdf (DataFrame): PySpark DataFrame representing user data. Should have columns 'userId' and 'featur_name'.

        Returns:
        - DataFrame: Updated PySpark DataFrame with filled missing users.
        """

        sdf_user_count = sdf.count()

        if sdf_user_count != self.unique_users_count:
            print(f"Missing Values: {self.unique_users_count - sdf_user_count}")
            missing_users = self.unique_users.select("userId").subtract(sdf.select("userId"))
            # Since the sdf is only two we rename the column based on sdf's second column
            missing_users_sdf = missing_users.withColumn(sdf.columns[1], lit(0))
            filled_missing_users = sdf.union(missing_users_sdf)

            return filled_missing_users
        else:
            return sdf


def build_features(spark_data):
        spark_data.build_song_counts()


def main():
    init_logging()

    # Create a Spark session with an appropriate app name and executor cores
    total_physical_cores = 16
    available_cores_for_spark = total_physical_cores - 2

    spark = (
        SparkSession.builder.appName("Sparkify")
        .config("spark.driver.memory", "12g")
        .config("spark.executor.memory", "12g")
        .config("spark.executor.cores", available_cores_for_spark)
        .getOrCreate()
    )
    sdf = spark.read.json("/Users/jacobfletcher/git/churn_project/data/mini_sparkify_event_data.json")

    spark_data = SparkData(sdf)

    spark_data.clean_sdf()
    build_features(spark_data)


   

    spark.stop()


if __name__ == "__main__":
    main()
