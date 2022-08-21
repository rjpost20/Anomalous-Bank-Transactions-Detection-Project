import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType


def set_weight_col(df, label_col, neg_class_weight):
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
    neg_class_weight : '`float` or 'balanced'
        New class weight to assign to negative class 
        (`0`) in weight column. If 'balanced', assigned 
        class weights will be equal to 1 - proportion of 
        class in dataframe. If `float`, negative class 
        will be assigned `neg_class_weight` and positive 
        class weights remain at 1.0.
    """
    if neg_class_weight == 'balanced':
        balancing_ratio = df.filter(F.col(label_col) == 1).count() / df.count()
        calculate_weights = F.udf(lambda x: balancing_ratio if x == 0 
                                  else 1.0 - balancing_ratio, DoubleType())
    else:
        calculate_weights = F.udf(lambda x: neg_class_weight if x == 0 
                                  else 1.0, DoubleType())

    df = df.withColumn('Weight', calculate_weights(label_col))

    return df


def spark_resample(df, ratio, new_count, class_field, 
                   pos_class, shuffle, random_state):
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

    oversample_fraction = (ratio * new_count) / total_pos
    undersample_fraction = ((1 - ratio) * new_count) / total_neg

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
