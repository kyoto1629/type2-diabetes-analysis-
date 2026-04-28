# Databricks notebook source
# MAGIC %md
# MAGIC Цель
# MAGIC
# MAGIC Провести исследовательский анализ данных (EDA) для выявления закономерностей и факторов, влияющих на развитие диабета.
# MAGIC
# MAGIC В рамках ноутбука:
# MAGIC
# MAGIC анализируется распределение признаков;
# MAGIC
# MAGIC исследуется целевая переменная;
# MAGIC
# MAGIC изучается влияние факторов (BMI, glucose, образ жизни);
# MAGIC
# MAGIC строятся базовые зависимости и агрегаты.

# COMMAND ----------

# DBTITLE 1,Импорт модулей
from pyspark.sql.functions import col, avg, count, when

# COMMAND ----------

# DBTITLE 1,Загрузка очищенных данных
df = spark.table("diabetes_cleaned")

# COMMAND ----------

# DBTITLE 1,Преобразуем категориальную переменную physical_activity
df = df.withColumn(
    "physical_activity_num",
    when(col("physical_activity") == "low", 1)
    .when(col("physical_activity") == "medium", 2)
    .when(col("physical_activity") == "high", 3)
)

# Преобразуем категориальную переменную physical_activity в числовую шкалу (low=1, medium=2, high=3) для упрощения агрегирования и дальнейшего анализа

# COMMAND ----------

# DBTITLE 1,Общая статистика
print("Количество строк:")
print(df.count())

print("Количество уникальных пациентов:")
df.select("patient_id").distinct().count()

# COMMAND ----------

# DBTITLE 1,Распределение целевой переменной
df.groupBy("future_diabetes_5yr").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC Распределение целевой переменной: примерно половина пациентов заболело диабетом и вторая половина здорова, с небольшим перевесом в сторону заболевших.

# COMMAND ----------

# DBTITLE 1,Анализ ключевых показателей здоровья
df.select(
    avg("BMI").alias("avg_BMI"),
    avg("fasting_glucose").alias("avg_glucose"),
    avg("HbA1c").alias("avg_HbA1c"),
    avg("blood_pressure").alias("avg_bp")
).show()

# COMMAND ----------

# DBTITLE 1,Сравнение групп риска
df.groupBy("future_diabetes_5yr").agg(
    avg("BMI").alias("avg_BMI"),
    avg("fasting_glucose").alias("avg_glucose"),
    avg("HbA1c").alias("avg_HbA1c"),
    avg("stress_level").alias("avg_stress"),
    avg("physical_activity_num").alias("avg_activity")
).show()

# COMMAND ----------

# DBTITLE 1,Анализ по возрасту
df.groupBy("age").count().orderBy("age").show()

# COMMAND ----------

# DBTITLE 1,Построение графика по возрасту

import matplotlib.pyplot as plt

# Группируем по возрасту
age_df = (
    df.groupBy("age")
      .count()
      .orderBy("age")
)

# переводим в pandas (для визуализации)
age_pd = age_df.toPandas()

plt.figure()
plt.plot(age_pd["age"], age_pd["count"])
plt.title("Распределение записей по возрасту")
plt.xlabel("Возраст")
plt.ylabel("Количество записей")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Количество записей увеличивается с возрастом пациента:
# MAGIC
# MAGIC минимальный возраст в данных — 19 лет
# MAGIC наблюдается постепенный рост числа записей с возрастом
# MAGIC максимальная плотность данных приходится на возраст 35–70 лет (по видимому распределению)
# MAGIC
# MAGIC -> это говорит о том, что датасет смещён в сторону более взрослой популяции

# COMMAND ----------

# DBTITLE 1,Анализ образа жизни
df.groupBy("physical_activity").agg(
    avg("future_diabetes_5yr").alias("risk_rate")
).show()

# COMMAND ----------

# DBTITLE 1,Построение графика по анализу образа жизни
lifestyle_df = (
    df.groupBy("physical_activity")
      .agg(
          avg("future_diabetes_5yr").alias("risk_rate")
      )
)

lifestyle_pd = lifestyle_df.toPandas()

plt.figure()
plt.bar(lifestyle_pd["physical_activity"], lifestyle_pd["risk_rate"])
plt.title("Риск диабета в зависимости от физической активности")
plt.xlabel("Физическая активность")
plt.ylabel("Средний риск диабета")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Можно наблюдать выраженную зависимость между уровнем физической активности и риском развития диабета.
# MAGIC
# MAGIC при низкой физической активности риск (risk_rate) составляет ~0.77;
# MAGIC при высокой физической активности риск составляет ~0.018;
# MAGIC
# MAGIC -> пациенты с низкой активностью демонстрируют существенно более высокий уровень риска по сравнению с активной группой.

# COMMAND ----------

# DBTITLE 1,Корреляционный анализ (упрощённый)
from pyspark.sql.functions import corr

df.select(
    corr("BMI", "future_diabetes_5yr").alias("BMI_corr"),
    corr("fasting_glucose", "future_diabetes_5yr").alias("glucose_corr"),
    corr("HbA1c", "future_diabetes_5yr").alias("hba1c_corr")
).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Наблюдается связь между метаболическими показателями и вероятностью развития диабета в течение 5 лет.

# COMMAND ----------

# MAGIC %md
# MAGIC Общий вывод
# MAGIC
# MAGIC Датасет сбалансирован по целевой переменной;
# MAGIC Метаболические показатели (glucose, HbA1c, BMI) имеют устойчивую связь с исходом заболевания;
# MAGIC Физическая активность демонстрирует сильную ассоциацию с риском;
# MAGIC Данные имеют выраженную медицинскую и поведенческую структуру;
# MAGIC Наблюдается логичная зависимость факторов риска от развития диабета.
# MAGIC
# MAGIC Итог для проекта
# MAGIC
# MAGIC EDA подтверждает, что:
# MAGIC -> данные подходят для построения модели прогнозирования диабета
# MAGIC -> ключевые факторы риска выделяются статистически
# MAGIC -> можно переходить к feature engineering и моделированию