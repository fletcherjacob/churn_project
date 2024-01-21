import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql import functions as F
from pyspark.sql.functions import col, from_unixtime, date_trunc, udf, lit
from datetime import datetime



class SparkData:
    def __init__(self, path, preproccessing_job= True):
        self.path = path
        self.init_logging()
        self.sdf = None
        self.create_spark_session()
        try:
            if self.path.endswith(".json"):
                self.sdf = self.spark.read.json(path)

            if self.path.endswith(".csv"):
                self.sdf = self.spark.read.csv(path, header = True)
        except Exception as e:
            print(e)
            self.app.error("Make Sure File is a valid format('.csv' or '.json')")
            exit()

        if preproccessing_job:
            self.preproccessing_run()

        self.unique_users = None
        self.unique_users_count = None

    def get_log_time(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def init_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.app = logging.getLogger("SPARKIFY")
        self.app.info("Logging initialized %s.", self.get_log_time())


    def clean_sdf(self):
        self.columns_to_keep = ["userId", "artist", "ts", "page", "level", "location"]
        self.sdf = self.sdf.select(self.columns_to_keep).dropna(subset="userId")
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

    def build_labels(self):
        self.labels = (
            self.sdf[["userId", "page"]]
            .filter(col("page").isin(["Cancellation Confirmation"]))
            .distinct()
        )
        self.labels = self.labels.withColumn("label", lit(1))
        self.labels = self.labels.drop("page")


        self.labels = self.handle_missing_users(self.labels,"label")


        return self

    def build_features(self):
        self.build_song_counts()
        self.build_user_levels()
        self.build_positive_usage()
        self.build_negative_usage()
        self.build_user_unique_locations()
        self.build_labels()

        dfs = [
            self.song_counts,
            # self.song_listened_mean,
            # self.average_daily_listens,
            self.user_level,
            self.positive_usage,
            self.neg_interactions,
            self.unique_locations,
            # self.distinct_artist,
            # self.page_count,
        ]

        self.processed_sdf = self.labels

        for df in dfs:
            self.processed_sdf = self.processed_sdf.join(df, "userId", "outer")
            del df

        return self


    def export_file(self, output_path, file_type):
        if file_type  == "csv":
            self.processed_sdf.coalesce(1).write.json(output_path, mode='overwrite')
        if file_type == 'json':
            self.processed_sdf.coalesce(1).write.json(output_path, mode='overwrite')
        else:
            logging.info(f"Please select eithe json or csv as filetype {str(file_type)} is not a suported format")

    def create_spark_session(self,app_name="Sparkify", default_settings = True ,  total_physical_cores=16,driver_memory = 8,executor_memory = 8):
    # Calculate available cores for Spark
        if default_settings == False:
            total_physical_cores = input(" Available Cores")
            driver_memory =  input(" Driver Memory Allowance")
            executor_memory = input("Executor Memory Allowance")

        available_cores_for_spark =int( total_physical_cores - 2)
        # Configure Spark session
        self.spark = (
            SparkSession.builder.appName(app_name)
            .config("spark.driver.memory", str(int(driver_memory)) + "g")
            .config("spark.executor.memory", str(int(executor_memory)) + "g")
            .config("spark.executor.cores", available_cores_for_spark)
            .getOrCreate()
        )

        return self
    
    def spark_stop(self):
         self.spark.stop()

    def preproccessing_run(self):
        self.clean_sdf()
        self.build_features()
        output_path = "/Users/jacobfletcher/git/churn_project/data/lg_model_features"#input("File Outpu Path:")
        file_type = "json"#input("File Type:")
        self.export_file(
            output_path=output_path,
            file_type=file_type
        )
    
if __name__ == "__main__":
    # Instantiate the class and run preprocessing
    #path = input("Filepath: ")
    path ="/Users/jacobfletcher/git/churn_project/data/lg_sparkify_event_data.json"
    spark_data = SparkData(path=path)
    
 