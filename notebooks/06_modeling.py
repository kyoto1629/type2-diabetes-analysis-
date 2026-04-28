# Databricks notebook source
# MAGIC %md
# MAGIC Цель
# MAGIC
# MAGIC Построить модель для предсказания вероятности развития диабета в течение 5 лет.
# MAGIC
# MAGIC В рамках ноутбука:
# MAGIC
# MAGIC формируется датасет признаков;
# MAGIC
# MAGIC обучается модель логистической регрессии;
# MAGIC
# MAGIC оценивается качество модели (ROC-AUC);
# MAGIC
# MAGIC проводится интерпретация признаков.

# COMMAND ----------

# DBTITLE 1,Импорт модулей
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# ML инструменты
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# Метрики
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

# DBTITLE 1,Загрузка данных (таблица с фичами из 04_feature_engineering)
df = spark.table("diabetes_features")

# COMMAND ----------

# DBTITLE 1,Выбор признаков
feature_cols = [
    "BMI",
    "fasting_glucose",
    "HbA1c",
    "lifestyle_score",
    "metabolic_risk_score",
    "age"
]

target = "future_diabetes_5yr"

# COMMAND ----------

# DBTITLE 1,Обработка категориальных признаков (gender)
indexer = StringIndexer(
    inputCol="gender",
    outputCol="gender_index"
)

# COMMAND ----------

# DBTITLE 1,Сборка вектора признаков
# Объединяем все признаки в один вектор

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

# COMMAND ----------

# DBTITLE 1,Нормализация
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features"
)

# COMMAND ----------

# DBTITLE 1,Модель
# Создаём модель логистической регрессии

lr = LogisticRegression(
    featuresCol="scaled_features",
    labelCol=target
)

# COMMAND ----------

# MAGIC %md
# MAGIC Базовая модель для бинарной классификации

# COMMAND ----------

# DBTITLE 1,Pipeline
pipeline = Pipeline(stages=[
    assembler,
    scaler,
    lr
])

# COMMAND ----------

# MAGIC %md
# MAGIC Автоматизируем весь процесс (очистка -> признаки -> модель)

# COMMAND ----------

# DBTITLE 1,Деление на train/test - чтобы честно оценить модель
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

# DBTITLE 1,Обучение модели
model = pipeline.fit(train_df)

# COMMAND ----------

# DBTITLE 1,Предсказания
predictions = model.transform(test_df)

# COMMAND ----------

# DBTITLE 1,Оценка модели
evaluator = BinaryClassificationEvaluator(
    labelCol=target,
    metricName="areaUnderROC"
)

roc_auc = evaluator.evaluate(predictions)

print(f"ROC-AUC: {roc_auc}")

# COMMAND ----------

# MAGIC %md
# MAGIC Модель показала высокое качество (ROC-AUC ≈ 0.97), что говорит о сильном предсказательном сигнале в данных. Так как это синтетический датасет - это нормально.

# COMMAND ----------

# DBTITLE 1,Важность признаков
# Достаём модель из pipeline

lr_model = model.stages[-1]

# Коэффициенты
coefficients = lr_model.coefficients

# Вывод
for name, coef in zip(feature_cols, coefficients):
    print(f"{name}: {coef}")

# COMMAND ----------

# MAGIC %md
# MAGIC Интерпретация коэффициентов модели
# MAGIC
# MAGIC В результате обучения модели логистической регрессии были получены следующие коэффициенты признаков.
# MAGIC
# MAGIC Наибольшее положительное влияние на вероятность развития диабета оказывают:
# MAGIC
# MAGIC fasting_glucose (~1.90);
# MAGIC
# MAGIC metabolic_risk_score (~1.86);
# MAGIC
# MAGIC BMI (~1.26).
# MAGIC
# MAGIC -> это означает, что при увеличении этих показателей вероятность развития диабета возрастает
# MAGIC
# MAGIC Умеренное положительное влияние наблюдается у:
# MAGIC
# MAGIC HbA1c (~0.78);
# MAGIC
# MAGIC age (~0.58).
# MAGIC
# MAGIC -> данные признаки также увеличивают риск, но их вклад менее выражен
# MAGIC
# MAGIC Отрицательное влияние имеет:
# MAGIC
# MAGIC lifestyle_score (~ -0.15)
# MAGIC
# MAGIC -> это означает, что более здоровый образ жизни (высокий lifestyle_score) снижает вероятность развития диабета
# MAGIC
# MAGIC Общий вывод
# MAGIC
# MAGIC Модель показывает, что ключевыми факторами риска являются:
# MAGIC
# MAGIC уровень глюкозы;
# MAGIC
# MAGIC общий метаболический риск;
# MAGIC
# MAGIC индекс массы тела.
# MAGIC
# MAGIC Также подтверждается, что образ жизни оказывает защитный эффект.