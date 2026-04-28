# единая точка загрузки данных

def load_table(spark, table_name: str):
    return spark.table(table_name)