# Databricks notebook source
# MAGIC %md
# MAGIC Цель
# MAGIC
# MAGIC Провести когортный анализ для изучения динамики развития диабета во времени.
# MAGIC
# MAGIC В рамках ноутбука:
# MAGIC
# MAGIC формируются когорты пациентов;
# MAGIC
# MAGIC анализируется развитие заболевания во времени;
# MAGIC
# MAGIC сравниваются различные группы пациентов.

# COMMAND ----------

# DBTITLE 1,Импорт модулей
from pyspark.sql.functions import col, expr, count

# COMMAND ----------

# DBTITLE 1,Загрузка данных, нужна таблица, где есть on_set_year
df = spark.table("diabetes_cleaned")

# COMMAND ----------

# DBTITLE 1,Формирование cohort
df = df.withColumn(
    "cohort_year",
    col("onset_year")
)

# Cohort = год первого диагноза

# COMMAND ----------

# DBTITLE 1,Фильтруем, так как есть 0 значения
df_cohort = df.filter(col("cohort_year").isNotNull())

# COMMAND ----------

# DBTITLE 1,Считаем “возраст когорты”
df_cohort = df_cohort.withColumn(
    "years_since_onset",
    col("year") - col("cohort_year")
)

# COMMAND ----------

# DBTITLE 1,Размер когорт
cohort_size = (
    df_cohort
    .groupBy("cohort_year")
    .agg({"patient_id": "count"})
    .withColumnRenamed("count(patient_id)", "cohort_size")
)

# COMMAND ----------

# DBTITLE 1,Retention / progression
cohort_progression = (
    df_cohort
    .groupBy("cohort_year", "years_since_onset")
    .agg(
        count("patient_id").alias("patients_count")
    )
) 
# Считаем, сколько пациентов “живы” или остаются в наблюдении

# COMMAND ----------

# DBTITLE 1,Нормализация
cohort_df = cohort_progression.join(cohort_size, "cohort_year")

cohort_df = cohort_df.withColumn(
    "retention_rate",
    col("patients_count") / col("cohort_size")
)

# COMMAND ----------

# MAGIC %md
# MAGIC Нормализация нужна, чтобы привести все признаки к одному масштабу
# MAGIC
# MAGIC Проблема без нормализации
# MAGIC
# MAGIC У нас есть признаки:
# MAGIC
# MAGIC glucose ≈ 100
# MAGIC
# MAGIC BMI ≈ 25
# MAGIC
# MAGIC HbA1c ≈ 5
# MAGIC
# MAGIC stress_level ≈ 1–10
# MAGIC
# MAGIC Если их сложить или подать в модель:
# MAGIC
# MAGIC модель начнёт “думать”, что большие числа важнее

# COMMAND ----------

# DBTITLE 1,Подготовка к визуализации
cohort_pd = cohort_df.toPandas()

# COMMAND ----------

# DBTITLE 1,Pivot (таблица когорт)
cohort_pivot = cohort_pd.pivot(
    index="cohort_year",
    columns="years_since_onset",
    values="retention_rate"
)

# COMMAND ----------

# DBTITLE 1,Визуализация
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(cohort_pivot)
plt.title("Cohort Analysis (Retention)")
plt.xlabel("Years since onset")
plt.ylabel("Cohort year")
plt.colorbar()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ось X -> years since onset (сколько лет прошло после диагноза)
# MAGIC
# MAGIC ось Y -> cohort_year (год постановки диагноза)
# MAGIC
# MAGIC цвет -> retention_rate (доля пациентов, оставшихся в данных)
# MAGIC
# MAGIC Видим естественное снижение retention со временем
# MAGIC
# MAGIC Это может означать:
# MAGIC
# MAGIC смертность;
# MAGIC
# MAGIC потерю наблюдений (dropout);
# MAGIC
# MAGIC окончание окна данных.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Вывод по cohort analysis
# MAGIC
# MAGIC Наблюдается ожидаемое снижение retention rate с увеличением времени после постановки диагноза.
# MAGIC
# MAGIC ранние когорты имеют более длинную историю наблюдений;
# MAGIC
# MAGIC поздние когорты ограничены временным окном данных;
# MAGIC
# MAGIC структура данных демонстрирует типичное “cohort decay” (затухание когорты со временем) поведение.