from pyspark.sql.functions import col


def clean_data(df):
    # убираем null в ключевых колонках
    df = df.filter(
        col("age").isNotNull() &
        col("BMI").isNotNull() &
        col("fasting_glucose").isNotNull()
    )
    
    return df