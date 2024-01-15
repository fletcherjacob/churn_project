import boto3
import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql import functions as F
from pyspark.sql.functions import col, from_unixtime, date_trunc, udf, lit


def init_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized.")


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
            self.sdf[["userId", "artist"]]
            .dropna(subset="artist")
            .groupBy("userId")
            .count()
        )
        self.song_counts = self.song_counts.withColumnRenamed("count", "song_counts")

        self.song_counts = self.handle_missing_users(self.song_counts, "song_counts")

    def build_user_levels(self):
        user_level = (
            self.sdf[["userId", "level", "ts"]]
            .orderBy("ts", ascending=False)
            .dropDuplicates(subset=["userId"])
            .select("userId", "level")
        )

        level_flag_udf = udf(lambda x: 1 if x == "paid" else 0, IntegerType())

        # one-hot encode
        self.user_level = user_level.withColumn(
            "level_flag", level_flag_udf(user_level["level"])
        ).select("userId", "level_flag")

        self.user_level = self.handle_missing_users(self.user_level, "user_level")

    def build_positive_usage(self):
        positive_usage_list = [
            "Thumbs Up",
            "Thumbs Down",
            "Add Friend",
            "Add to playlist",
        ]

        self.positive_usage = (
            self.sdf[["userId", "page"]]
            .filter(col("page").isin(positive_usage_list))
            .groupBy("userId")
            .count()
        )

        self.positive_usage = self.positive_usage.withColumnRenamed(
            "count", "pos_interactions"
        )

        self.positive_usage = self.handle_missing_users(
            self.positive_usage, "positive_interactions"
        )

    def build_negative_usage(self):
        neg_interactions_list = ["Error", "Help"]
        self.neg_interactions = (
            self.sdf[["userId", "page"]]
            .filter(col("page").isin(neg_interactions_list))
            .groupBy("userId")
            .count()
        )

        self.neg_interactions = self.neg_interactions.withColumnRenamed(
            "count", "_interactions"
        )

        self.neg_interactions = self.handle_missing_users(
            self.neg_interactions, "negative_interactions"
        )

    def build_user_unique_locations(self):
        self.unique_locations = (
            self.sdf.filter(self.sdf["location"].isNotNull())
            .groupBy("userId")
            .agg(F.countDistinct("location").alias("unique_locations"))
        )

        self.unique_locations = self.handle_missing_users(
            self.unique_locations, "unique_user_locations"
        )

    def handle_missing_users(self, sdf: DataFrame, feature_name) -> DataFrame:
        """
        Handle missing users in a PySpark DataFrame.

        Parameters:
        - sdf (DataFrame): PySpark DataFrame representing user data. Should have columns 'userId' and 'featur_name'.

        Returns:
        - DataFrame: Updated PySpark DataFrame with filled missing users.
        """

        sdf_user_count = sdf.count()

        if sdf_user_count != self.unique_users_count:
            logging.info(
                f"Missing Users {feature_name}: {self.unique_users_count - sdf_user_count}"
            )
            missing_users = self.unique_users.select("userId").subtract(
                sdf.select("userId")
            )
            # Since the sdf is only two we rename the column based on sdf's second column
            missing_users_sdf = missing_users.withColumn(sdf.columns[1], lit(0))
            filled_missing_users = sdf.union(missing_users_sdf)

            return filled_missing_users
        else:
            logging.info(
                f"Missing Users {feature_name}: {self.unique_users_count - sdf_user_count}"
            )
            return sdf

    def build_features(self):
        self.build_song_counts()
        self.build_user_levels()
        self.build_positive_usage()
        self.build_negative_usage()
        self.build_user_unique_locations()


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

    # Load Data To Spark Dataframe
    try:
        path = input("Provide location of json data:")
        "/Users/jacobfletcher/git/churn_project/data/mini_sparkify_event_data.json"
        sdf = spark.read.json(path)
    except Exception as e:
        logging.error(f"Error reading JSON file: {e}")

    spark_data = SparkData(sdf)
    spark_data.clean_sdf()
    spark_data.build_features()

    spark.stop()


if __name__ == "__main__":
    main()
