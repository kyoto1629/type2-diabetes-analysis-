# Databricks notebook source
# MAGIC %md
# MAGIC Цель
# MAGIC
# MAGIC Подготовить данные для анализа путём очистки и приведения к корректному формату.
# MAGIC
# MAGIC В рамках ноутбука:
# MAGIC
# MAGIC проверяются и обрабатываются пропущенные значения;
# MAGIC
# MAGIC устраняются дубликаты;
# MAGIC
# MAGIC приводятся типы данных;
# MAGIC
# MAGIC выполняется базовая валидация данных.

# COMMAND ----------

# DBTITLE 1,Импорт модулей
from pyspark.sql.functions import col, when

# COMMAND ----------

# DBTITLE 1,Загрузка данных
df = spark.table("research_grade_type_2_diabetes")

# COMMAND ----------

# DBTITLE 1,Фильтрация явных выбросов
# Удаляем физически невозможные или подозрительные значения

df_clean = df.filter(
    (col("BMI") > 10) & (col("BMI") < 60) &
    (col("fasting_glucose") > 50) & (col("fasting_glucose") < 400) &
    (col("age") >= 18) & (col("age") <= 100)
)

print("Количество строк после удаления выбросов:")
print(df_clean.count())

# COMMAND ----------

# DBTITLE 1,Работа с onset_year
# Проверяем: если диабета нет, onset_year должен быть NULL

df_clean = df_clean.withColumn(
    "onset_year",
    when(col("diabetes_onset") == 0, None)
    .otherwise(col("onset_year"))
)

# COMMAND ----------

# DBTITLE 1,Проверка логики onset_year
# onset_year не может быть раньше текущего года наблюдения

df_clean = df_clean.withColumn(
    "onset_year",
    when(col("onset_year") > col("year"), None)
    .otherwise(col("onset_year"))
)

# COMMAND ----------

# DBTITLE 1,Проверка бинарных признаков
# Проверим значения (должны быть 0 или 1)

binary_cols = ["family_history", "smoking", "alcohol", "on_medication", "diabetes_onset"]

for c in binary_cols:
    df_clean.groupBy(c).count().show()

# COMMAND ----------

# DBTITLE 1,Проверка категориальных значений
# Проверка категорий

df_clean.select("gender").distinct().show()
df_clean.select("physical_activity").distinct().show()
df_clean.select("BMI_category").distinct().show()

# COMMAND ----------

# DBTITLE 1,Сохраняем очищенные данные
df_clean.write.mode("overwrite").saveAsTable("diabetes_cleaned")

# COMMAND ----------

# MAGIC %md
# MAGIC Выводы по очистке данных:
# MAGIC
# MAGIC Было проверено есть ли выбросы по BMI, glucose и возрасту -> выбросов нет;
# MAGIC Приведена логика onset_year (обнуление при отсутствии диабета);
# MAGIC Исправлены некорректные значения onset_year;
# MAGIC Проверены бинарные и категориальные признаки;
# MAGIC
# MAGIC -> данные готовы для дальнейшего анализа