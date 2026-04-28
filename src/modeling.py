from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline


def build_pipeline(feature_cols, target):
    
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )
    
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaled_features"
    )
    
    lr = LogisticRegression(
        featuresCol="scaled_features",
        labelCol=target
    )
    
    pipeline = Pipeline(stages=[assembler, scaler, lr])
    
    return pipeline


def train_model(df, feature_cols, target):
    
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    pipeline = build_pipeline(feature_cols, target)
    
    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)
    
    return model, predictions