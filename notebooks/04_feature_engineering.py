# Databricks notebook source
# MAGIC %md
# MAGIC Цель ноутбука
# MAGIC
# MAGIC Превратить “сырые признаки” в:
# MAGIC
# MAGIC более информативные фичи;
# MAGIC сегменты пользователей;
# MAGIC признаки риска;
# MAGIC признаки взаимодействий.

# COMMAND ----------

# DBTITLE 1,Импорт модулей
from pyspark.sql.functions import col, when, round

# COMMAND ----------

# DBTITLE 1,Загрузка данных
df = spark.table("diabetes_cleaned")

# COMMAND ----------

# DBTITLE 1,Добавление колонки с категоризацией индекса массы тела
df = df.withColumn(
    "BMI_group",
    when(col("BMI") < 18.5, "underweight")
    .when((col("BMI") >= 18.5) & (col("BMI") < 25), "normal")
    .when((col("BMI") >= 25) & (col("BMI") < 30), "overweight")
    .otherwise("obese")
)

# COMMAND ----------

# DBTITLE 1,Возрастные группы
df = df.withColumn(
    "age_group",
    when(col("age") < 30, "young")
    .when((col("age") >= 30) & (col("age") < 45), "adult")
    .when((col("age") >= 45) & (col("age") < 60), "middle")
    .otherwise("senior")
)

# COMMAND ----------

# DBTITLE 1,Индекс метаболического риска
df = df.withColumn(
    "metabolic_risk_score",
    round(
        col("BMI") * 0.3 +
        col("fasting_glucose") * 0.4 +
        col("HbA1c") * 0.3,
        2
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC Это простой агрегированный показатель, который объединяет несколько коррелирующих признаков в один, чтобы упростить модель и потенциально усилить предсказательную способность.
# MAGIC
# MAGIC Веса выбраны эвристически (на глаз)
# MAGIC
# MAGIC Но с логикой:
# MAGIC
# MAGIC glucose -> 0.4 (самый важный фактор)
# MAGIC
# MAGIC BMI -> 0.3 (значимый, но косвенный)
# MAGIC
# MAGIC HbA1c -> 0.3 (тоже важный, но сглаженный показатель)
# MAGIC

# COMMAND ----------

df = df.withColumn(
    "physical_activity_num",
    when(col("physical_activity") == "low", 1)
    .when(col("physical_activity") == "medium", 2)
    .when(col("physical_activity") == "high", 3)
)

# COMMAND ----------

# DBTITLE 1,Lifestyle score
df = df.withColumn(
    "lifestyle_score",
    col("physical_activity_num") * 0.4 +
    (10 - col("stress_level")) * 0.3 +
    col("sleep_hours") * 0.3
)

# COMMAND ----------

# MAGIC %md
# MAGIC Признак lifestyle_score агрегирует поведенческие факторы в единую метрику, это позволяет нам упростить анализ и выявить общее влияние образа жизни на риск диабета.
# MAGIC
# MAGIC Признак stress_level:
# MAGIC
# MAGIC stress_level от 1 до 10
# MAGIC где 10 = плохо, 1 = хорошо
# MAGIC
# MAGIC поэтому я перевернул шкалу в формуле (10 - stress_level) 
# MAGIC
# MAGIC -> теперь чем больше - тем лучше

# COMMAND ----------

# DBTITLE 1,Interaction features
df = df.withColumn(
    "glucose_BMI_interaction",
    col("fasting_glucose") * col("BMI")
)

# COMMAND ----------

# MAGIC %md
# MAGIC Создаем комбинацию факторов риска

# COMMAND ----------

# DBTITLE 1,Risk segmentation
df = df.withColumn(
    "risk_segment",
    when(col("metabolic_risk_score") < 100, "low")
    .when(col("metabolic_risk_score") < 150, "medium")
    .otherwise("high")
)

# COMMAND ----------

# MAGIC %md
# MAGIC Cоздаем признак с категориальным значеним (сегментируем колоку "metabolic_risk_score")

# COMMAND ----------

# DBTITLE 1,Финальный набор фичей
selected_features = df.select(
    # идентификаторы
    "patient_id",
    "year",

    # raw признаки
    "age",
    "BMI",
    "fasting_glucose",
    "HbA1c",

    # категории 
    "age_group",
    "BMI_group",

    # engineered признаки
    "metabolic_risk_score",
    "lifestyle_score",
    "glucose_BMI_interaction",
    "risk_segment",

    # target
    "future_diabetes_5yr"
)

# COMMAND ----------

selected_features.write.mode("overwrite").saveAsTable("diabetes_features")

# COMMAND ----------

# MAGIC %md
# MAGIC Вывод по 04_feature engineering
# MAGIC
# MAGIC В рамках данного этапа были созданы дополнительные признаки для улучшения качества анализа и подготовки данных к моделированию.
# MAGIC
# MAGIC Проведена сегментация пациентов по возрасту (age_group) и BMI (BMI_group)
# MAGIC
# MAGIC Сформирован агрегированный показатель метаболического риска (metabolic_risk_score) на основе ключевых медицинских признаков
# MAGIC
# MAGIC Добавлен показатель образа жизни (lifestyle_score), учитывающий физическую активность, стресс и сон
# MAGIC
# MAGIC Создан interaction-признак (glucose_BMI_interaction) для учёта совместного влияния факторов
# MAGIC
# MAGIC Выполнена сегментация пользователей по уровню риска (risk_segment)
# MAGIC
# MAGIC -> в результате получен расширенный набор признаков, отражающий как медицинские, так и поведенческие аспекты
# MAGIC
# MAGIC -> данные подготовлены для дальнейшего анализа и построения моделей машинного обучения