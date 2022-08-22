import numpy as np
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import TrainValidationSplit
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def set_weight_col(df, label_col, pos_class_weight, neg_class_weight):
    """
    Calculates and creates a column of class weights 
    in a PySpark dataframe with an imbalanced binary 
    target class distribution.

    Parameters
    ----------
    df : Spark `DataFrame`
        The Spark `DataFrame` to assign the 
        `ClassWeight` column to
    label_col : Spark `Column`
        Label column name
    pos_class_weight : `float` or 'balanced'
        Class weight to assign to positive class 
        (`1`) in weight column. If 'balanced', assigned 
        class weights will be equal to 1 - proportion of 
        class in dataframe. If `float`, positive class 
        will be assigned `pos_class_weight`.
    neg_class_weight : '`float` or 'balanced'
        Class weight to assign to negative class 
        (`0`) in weight column. If 'balanced', assigned 
        class weights will be equal to 1 - proportion of 
        class in dataframe. If `float`, negative class 
        will be assigned `neg_class_weight`.
    """
    if pos_class_weight == 'balanced' or neg_class_weight == 'balanced':
        balancing_ratio = df.filter(F.col(label_col) == 1).count() / df.count()
        calculate_weights = F.udf(lambda x: balancing_ratio if x == 0 
                                  else 1.0 - balancing_ratio, DoubleType())
    else:
        calculate_weights = F.udf(lambda x: pos_class_weight if x == 1 
                                  else neg_class_weight, DoubleType())

    df = df.withColumn('Weight', calculate_weights(label_col))

    return df



def spark_resample(df, class_field, pos_class, shuffle, random_state,
                   ratio=None, new_count=None,
                   oversample_fraction=None,
                   undersample_fraction=None):
    """
    Resamples PySpark dataframe with an imbalanced binary 
    target class distribution through a combination of 
    undersampling the majority (negative) class and 
    oversampling the minority (positive) class.

    Parameters
    ----------
    df : Spark `DataFrame`
        The Spark `DataFrame` to resample
    ratio : `float`
        Desired ratio of positive (oversampled) class to 
        negative (undersampled) class in new `DataFrame`
    new_count : `int`
        Desired total count of observations (rows) in new
        `DataFrame`
    oversample_fraction : `float`
        Fraction to oversample minority class
    undersample_fraction : `float`
        Fraction to undersample majority class
    class_field : Spark `Column`
        Name of `Column` in `DataFrame` to resample
    pos_class : `DataType` of `class_field`
        Label of positive class in `class_field`
    shuffle : `bool`
        Whether to shuffle `DataFrame` before returning 
        resampled `DataFrame`. If `False`, positive 
        class observations will all be distributed on 
        the bottom of the resampled `DataFrame`.
    random_state : `int`
        Random seed for reproducibility
    """
    pos = df.filter(F.col(class_field)==pos_class)
    neg = df.filter(F.col(class_field)!=pos_class)

    total_pos = pos.count()
    total_neg = neg.count()
    
    if ratio != None and new_count != None:
        oversample_fraction = (ratio * new_count) / total_pos
        undersample_fraction = ((1 - ratio) * new_count) / total_neg
    elif oversample_fraction != None and undersample_fraction != None:
        oversample_fraction=oversample_fraction
        undersample_fraction=undersample_fraction

    pos_oversampled = pos.sample(withReplacement=True, 
                                 fraction=oversample_fraction, 
                                 seed=random_state)
    
    neg_undersampled = neg.sample(withReplacement=False, 
                                  fraction=undersample_fraction, 
                                  seed=random_state)

    resampled_df = neg_undersampled.union(pos_oversampled)

    if shuffle == True:
        resampled_df = resampled_df.withColumn('rand', 
                                               F.rand(seed=42))\
                                               .orderBy('rand')

    return resampled_df.drop('rand')



def grid_search(stages_with_classifier, train_df,
                model_grid, parallelism, seed=42):
    """
    Grid search over a PySpark classifier by passing in
    a `stages_with_classifier` object, a training
    dataframe, and a model parameter grid.

    Parameters
    ----------
    stages_with_classifier : `list`
        List containing Spark preprocessing pipeline with
        classifier
    df : Spark `DataFrame`
        The Spark training `DataFrame` to train the
        TrainValidationSplit model with
    model_grid : `list`
        ParamGridBuilder specifying parameters to tune
    parallelism : `int`
        The number of threads to use when running parallel
        algorithms
    seed : `int`
        Random seed for reproducibility
    """
    pipeline = Pipeline(stages=stages_with_classifier)

    evaluator = BinaryClassificationEvaluator(labelCol='Anomalous', 
                                              metricName="areaUnderPR")

    tvs = TrainValidationSplit(estimator=pipeline, 
                               estimatorParamMaps=model_grid, 
                               evaluator=evaluator, 
                               parallelism=parallelism, 
                               seed=seed)

    tvs_model = tvs.fit(train_df)

    best_model_parameters = model_grid[np.argmax(tvs_model.validationMetrics)]

    print(best_model_parameters)

    return tvs_model



def score_model(model, train_df, test_df):
    """
    Prints Sklearn classification report of trained model
    for both training and testing data.

    Parameters
    ----------
    model : `PipelineModel` or `TrainValidationSplitModel`
        Model to score
    train_df : Spark `DataFrame`
        The Spark `DataFrame` to train the `model` on
    test_df : Spark `DataFrame`
        The Spark `DataFrame` to test the `model` on
    """
    try:
        print(str(model.bestModel.stages[-1]).split(':')[0])
        length = len(str(model.bestModel.stages[-1]).split(':')[0])
    except AttributeError:
        print(str(model.stages[-1]).split(':')[0])
        length = len(str(model.stages[-1]).split(':')[0])
    print('-'*length, '\n')

    train_pred = model.transform(train_df)
    test_pred = model.transform(test_df)

    y_true_train = np.array(train_pred.select('Anomalous').collect())
    y_true_test = np.array(test_pred.select('Anomalous').collect())

    y_predicted_train = np.array(train_pred.select('prediction').collect())
    y_predicted_test = np.array(test_pred.select('prediction').collect())

    target_names = ['0: Non-anomalous', '1: Anomalous']

    print('Training data:\n')
    print(classification_report(y_true_train, 
                                y_predicted_train, 
                                target_names=target_names, 
                                digits=3))
    print('-'*60, '\n')

    print('Testing data:\n')
    print(classification_report(y_true_test, 
                                y_predicted_test, 
                                target_names=target_names, 
                                digits=3))



def plot_confusion_matrix(model, test_df):
    """
    Plots Sklearn confusion matrix using fit model on
    testing data.

    Parameters
    ----------
    model : `PipelineModel` or `TrainValidationSplitModel`
        Model to score
    test_df : Spark `DataFrame`
        The Spark `DataFrame` to display confusion matrix
        scores of with the `model`
    """
    y_true = np.array(test_df.select('Anomalous').collect())
    y_predicted = np.array(model.transform(test_df).select('prediction').collect())
    
    fig, ax1 = plt.subplots(figsize=(18, 7))

    labels = ['0: Non-anomalous', '1: Anomalous']
    cm = confusion_matrix(y_true, y_predicted)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    cm_disp.plot(cmap='Reds', ax=ax1, values_format='')
    ax1.set_title('Confusion Matrix - Test Data', size=15, fontweight='bold')
    ax1.set_xlabel(ax1.get_xlabel(), size=13)
    ax1.set_ylabel(ax1.get_ylabel(), size=13)
    ax1.tick_params(axis='both', which='both', labelsize=12)
    for labels in cm_disp.text_.ravel():
        labels.set_fontsize(12)
    ax1.grid(False)

