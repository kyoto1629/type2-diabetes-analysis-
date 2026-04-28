# Databricks notebook source
# MAGIC %md
# MAGIC Цель ноутбука
# MAGIC
# MAGIC Загрузить исходный датасет в среду Databricks и создать таблицу для дальнейшего анализа.
# MAGIC
# MAGIC В рамках ноутбука:
# MAGIC
# MAGIC выполняется загрузка CSV-файла;
# MAGIC
# MAGIC создаётся таблица в Spark;
# MAGIC
# MAGIC проводится первичная проверка данных.

# COMMAND ----------

# DBTITLE 1,Импорт модулей
from pyspark.sql.functions import col, sum, countDistinct, min, max

# COMMAND ----------

# DBTITLE 1,Загружаем таблицу из Databricks
df = spark.table("research_grade_type_2_diabetes")

# COMMAND ----------

# DBTITLE 1,Первичный осмотр данных
# Вывод схемы (типы данных)
print("Схема данных:")
df.printSchema()

# Просмотр первых 10 строк
print("Первые 10 записей:")
display(df.limit(10))

# COMMAND ----------

# DBTITLE 1,Базовая информация о датасете
# Общее количество строк
total_rows = df.count()

# Количество уникальных пациентов
unique_patients = df.select(countDistinct("patient_id")).collect()[0][0]

print(f"Общее количество строк: {total_rows}")
print(f"Количество уникальных пациентов: {unique_patients}")

# COMMAND ----------

# DBTITLE 1,Анализ временного диапазона
# Минимальный и максимальный год
year_stats = df.select(min("year").alias("min_year"),
                       max("year").alias("max_year"))

print("Диапазон лет:")
display(year_stats)

# Проверка распределения данных по годам
print("Количество записей по годам:")
display(
    df.groupBy("year")
      .count()
      .orderBy("year")
)

# COMMAND ----------

# DBTITLE 1,Проверка пропущенных значений
# Считаем количество NULL значений по каждому столбцу
null_counts = df.select([
    sum(col(c).isNull().cast("int")).alias(c)
    for c in df.columns
])

print("Количество пропусков по столбцам:")
display(null_counts)

# COMMAND ----------

# DBTITLE 1,Проверка дубликатов
# Проверяем дубликаты по колонкам (patient_id + year)
duplicates = (
    df.groupBy("patient_id", "year")
      .count()
      .filter(col("count") > 1)
)

duplicates_count = duplicates.count()

print(f"Количество дубликатов (patient_id + year): {duplicates_count}")

# COMMAND ----------

# DBTITLE 1,Базовые статистики по числовым признакам
print("Базовая статистика по числовым признакам:")
display(df.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC Выводы по первичному анализу данных
# MAGIC 1. Общая информация
# MAGIC Датасет содержит 496,362 записей и 10,000 уникальных пациентов;
# MAGIC Данные имеют временную структуру (наблюдения по годам);
# MAGIC В среднем ~50 записей на одного пациента.
# MAGIC
# MAGIC 2. Временной диапазон
# MAGIC Период: 1976 – 2026;
# MAGIC До 2006 года по 10,000 записей ежегодно;
# MAGIC После 2006 наблюдается снижение количества записей.
# MAGIC -> вероятно связано с выбыванием пациентов (смертность)
# MAGIC
# MAGIC 3. Качество данных
# MAGIC Пропуски:
# MAGIC Почти отсутствуют onset_year содержит много пропусков-> это ожидаемо, так как диабет есть не у всех
# MAGIC
# MAGIC Дубликаты:
# MAGIC Не обнаружены
# MAGIC -> данные уникальны по (patient_id, year)
# MAGIC
# MAGIC 4. Основные признаки
# MAGIC Медицинские показатели (glucose, HbA1c, insulin и др.);
# MAGIC Демография (возраст, пол, страна);
# MAGIC Поведенческие факторы (сон, питание, активность);
# MAGIC Целевая переменная: future_diabetes_5yr.
# MAGIC
# MAGIC 5. Первичные наблюдения
# MAGIC Средний BMI ≈ 26 -> зона overweight;
# MAGIC Средний возраст ≈ 54 года;
# MAGIC Показатели здоровья имеют широкий диапазон.
# MAGIC -> данные подходят для анализа и моделирования
# MAGIC
# MAGIC 6. Итог
# MAGIC Датасет чистый и хорошо структурирован;
# MAGIC Есть временная зависимость признаков;
# MAGIC Можно проводить анализ прогрессии заболевания и строить прогнозы.
# MAGIC 7. Следующий шаг
# MAGIC Очистка данных;
# MAGIC Проверка выбросов;
# MAGIC Подготовка признаков для анализа и моделей.

# COMMAND ----------

