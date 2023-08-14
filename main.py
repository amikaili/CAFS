import numpy as np
import pandas as pd
import xarray
from datetime import datetime, date
from pyspark import mllib
from pyspark import dataframe
#from pyspark import spark
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, vectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import cafs.libraries.logging as cafs_logging

def detected_clouds(s3_keys: list[str]) -> str:

    store = CafsObjectStore()
    logger.info("Detecting clouds on %s", s3_keys)
    the_path = Path(s3_keys[0])
    local_path = Path("/tmp") / the_path.name
    store.download_file(local_path, s3_keys[0])
    ds = xarray.open_dataset(local_path)
    df = spark.createDataFrame(ds)

    x_size = ds.x.shape[0]
    y_size = ds.y.shape[0]
    n_areas = 12

    # Prepare Data for Machine Learning
    categoricalColumns = ['pixel', 'cloudMask']
    stages = []
    for categoricalCol in categoricalColumns:
        StringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
        encoder = OneHotEncoderEstimator(inputCol = [stringIndexer.getOutputCol()], outputCols = [categoricalCol + "classvec"])
        stages += [stringIndexer, encoder]

    label_stringIdx = StringIndexer(inputCol = 'pixel', outputCol='label')
    stages += [label_stringIdx]
    numericCols = ['cloudMask']
    assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCols="features")
    stages += [assembler]

    # Create a Pipeline, to chain multiple workflow of Transformers and Estimators
    pipeline = Pipeline(stages = stages)
    pipelineModel = pipeline.fit(df)
    df = pipelineModel.transform(df)
    cols = df.columns
    selectedCols = ['label', 'features'] + cols
    df = df.select(selectedCols)
   
    df.printSchema()

    # To Pandas dataframe, Transpose dataframe
    pd.DataFrame(df.take(4), columns=df.columns).Transpose
    numeric_features = [t[0] for t in df.dtypes if t[1] == 'int']
    df.select(numeric_features).describe().toPandas().transpose()

    # Split training data for LogisticRegression Model
    train, test = df.randomSplit([0.7, 0.3], seed = 2023)

    print("Training Dataset Count: " + str(train.count()))
    print("Test Dataset Count: " + str(test.count()))

    # Create LogisticRegression Model
    lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
    lrModel = lr.fit(train)
    beta = np.sort(lrModel.coefficients)
    plt.plot(beta)
    plt.ylabel('Beta Coefficients')
    plt.show()

    # TODO: Calculate training set validation METRICS like AreaUnderRock
    #    and receiver-operating-characteristics (ROC Curve), Precision, recall...

    # Prediction from learned model from training it
    predictions = lrModel.transform(test)
    predictions.select('pixel', 'cloudMask').show()

    # Evaluate pridications:
    evaluator = BinaryClassificationEvaluator()
    print('Test Area under ROC', evaluator.evaluate(predications))

    # TODO: Bench mark above LogisticRegression model, against other ML models...
    #        e.g. DecisionTreeClassifier (for better categorial features handling)
    #              or RandomForestClassifier
   
    return str(df)


