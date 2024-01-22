import logging
from pyspark.sql import functions as F
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    accuracy_score,
    fbeta_score,
    roc_auc_score,
)


def create_spark_session(
    app_name="Sparkify",
    default_settings=True,
    total_physical_cores=16,
    driver_memory=8,
    executor_memory=8,
    quiet=True,
):
    try:
        spark.shutdown()
    except Exception as e:
        print(e)

    if default_settings == False:
        total_physical_cores = input(" Available Cores")
        driver_memory = input(" Driver Memory Allowance")
        executor_memory = input("Executor Memory Allowance")

    available_cores_for_spark = int(total_physical_cores - 2)
    # Configure Spark session
    spark = (
        SparkSession.builder.appName(app_name)
        # .config("spark.driver.memory", str(int(driver_memory)) + "g")
        # .config("spark.executor.memory", str(int(executor_memory)) + "g")
        .config("spark.memory.fraction", "0.8")
        .config("spark.executor.cores", available_cores_for_spark)
        .getOrCreate()
    )

    def quiet_logs(sc):
        logger = sc._jvm.org.apache.log4j
        logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
        logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
        logger.LogManager.getLogger("agerMaster").setLevel(logger.Level.ERROR)
        logger.LogManager.getLogger("Thread-2").setLevel(logger.Level.ERROR)

    if quiet == True:
        quiet_logs(spark.sparkContext)

    return spark


def load_dataframe(file_path, spark, feature_cols=None, label="label"):
    # Start Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    app = logging.getLogger("SPARKIFY")
    app.info("Logging initialized %s.", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Determine which file to load
    if file_path.endswith(".csv"):
        df = spark.read.option("header", "True").csv(file_path)
    elif file_path.endswith(".json"):
        df = spark.read.option("header", "True").json(file_path)
    else:
        raise ValueError("Unsupported file format. Supported formats: CSV, JSON")

    # Drop unselected features
    if feature_cols:
        for col_name in df.columns:
            if col_name not in feature_cols and col_name != label:
                app.warning(
                    f"{col_name} feature is being dropped : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                df = df.drop(col_name)

    # Cast all features as double, *Note Specific To Project Use Case, discretion if used outside of project
    df = df.select(*(col(c).cast("double").alias(c) for c in df.columns))

    return df, app


def score_metrics(df,y_true_col,y_pred_col):
    labels_list = df.select(y_true_col).collect()
    predictions_list = df.select(y_pred_col).collect()
    y_test = [row[y_true_col] for row in labels_list]
    y_pred = [row[y_pred_col] for row in predictions_list]

    def specificity_score(y_test, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        return tn / (tn + fp)

    specificity = specificity_score(y_test, y_pred)
    f1 = f1_score(y_pred=y_pred, y_true=y_test)
    precision = precision_score(y_pred=y_pred, y_true=y_test)
    recall = recall_score(y_pred=y_pred, y_true=y_test)
    accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
    f2 = fbeta_score(y_pred=y_pred, y_true=y_test, beta=2)
    auc = roc_auc_score(y_score=y_pred, y_true=y_test)

    metrics = {
        "AUC": auc,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "accuracy": accuracy,
        "f2": f2,
        "specificity": specificity,


    }
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    return metrics






