from pyspark.sql.functions import col, when


def add_features(df):
    
    # lifestyle_score (пример)
    df = df.withColumn(
        "lifestyle_score",
        col("sleep_hours") - col("stress_level")
    )
    
    # metabolic_risk_score (упрощённо)
    df = df.withColumn(
        "metabolic_risk_score",
        col("BMI") + col("fasting_glucose") + col("HbA1c")
    )
    
    # сегментация
    df = df.withColumn(
        "risk_segment",
        when(col("metabolic_risk_score") < 100, "low")
        .when(col("metabolic_risk_score") < 150, "medium")
        .otherwise("high")
    )
    
    return df