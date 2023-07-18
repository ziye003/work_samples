# Databricks notebook source
pip install s3fs

# COMMAND ----------

pip install matplotlib_venn

# COMMAND ----------

# import libraries
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats
import statistics
#import warnings
import scipy
import scipy.stats as stats
from IPython.display import set_matplotlib_formats

# COMMAND ----------

# set figure resolution
plt.rcParams['figure.dpi']=150
%config InlineBackend.figure_format = 'png2x'
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png2x')

# COMMAND ----------

# import spark libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as fx
from pyspark.sql.window import Window
from pyspark.sql.functions import split,concat,col,lit,stddev_pop,rand


from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.mllib.evaluation import BinaryClassificationMetrics

# import sklearn libraries
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc,roc_curve

# COMMAND ----------

# MAGIC %sql
# MAGIC use interbio_followup_HC

# COMMAND ----------

# MAGIC %md # load data from hive

# COMMAND ----------

filtered_train_df=spark.read.table('filtered_train')
filtered_train_df.limit(4).show()

# COMMAND ----------

print(len(filtered_train_df.columns),filtered_train_df.count())

# COMMAND ----------

# MAGIC %md # change long to wide format

# COMMAND ----------

name_columns = [col for col in filtered_train_df.columns if 'feature_label' not in col and 'intensity' not in col]
wide_filtered_train_df=filtered_train_df.drop('intensity').groupBy(name_columns).pivot('feature_label').agg({"intensity_fill_log_scaled":"max"})
wide_filtered_train_df.select(wide_filtered_train_df.columns[:10]).limit(4).show()

# COMMAND ----------

print(len(wide_filtered_train_df.columns),wide_filtered_train_df.count())

# COMMAND ----------

# MAGIC %md # define the # of features need to be selected:

# COMMAND ----------

K=20

# COMMAND ----------

# MAGIC %md #defind the disease/subtype of insterest

# COMMAND ----------

HC_subtype='HC5'

# COMMAND ----------

# Prepare your data
mtb_col=[col for col in wide_filtered_train_df.columns if 'mtb' in col]
assembler = VectorAssembler(inputCols=mtb_col, outputCol="features")
train_assembled = assembler.transform(wide_filtered_train_df)
train_df = train_assembled.select('features',HC_subtype)

# COMMAND ----------

# MAGIC %md # select the top K features using RF

# COMMAND ----------

# Train the Random Forest model
rf_model = RandomForestClassifier(featuresCol='features', labelCol=HC_subtype,numTrees=100, maxDepth=15, seed=42)
rf_fitted_model=rf_model.fit(train_df)

# COMMAND ----------

# Obtain feature importances
importances = rf_fitted_model.featureImportances
# Convert feature importances to a Python list
importance_list = importances.toArray().tolist()

# COMMAND ----------


# Create a dictionary of feature names and importances
feature_importances = dict(zip(mtb_col, importance_list))

# Sort the feature importances by importance values in descending order
sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

# COMMAND ----------

import matplotlib.pyplot as plt

# Obtain the feature importances from your model
feature_importances = rf_fitted_model.featureImportances

# Plot the histogram of feature importances
plt.hist(feature_importances, bins=20)
plt.xlabel('RF Feature Importance for {}'.format(HC_subtype))
plt.ylabel('Count')
plt.title('Histogram of Feature Importance for {}'.format(HC_subtype))
plt.show()

# COMMAND ----------

# Select the top-k most important features
selected_features = [feature for feature, importance in sorted_importances[:K]]

# Print the selected features
print(len(selected_features))

# COMMAND ----------

# Subset the train DataFrame with the selected features
rf_train_df = filtered_train_df.filter(filtered_train_df.feature_label.isin(selected_features))

# COMMAND ----------

rf_train_df.show(5)

# COMMAND ----------

# MAGIC %md ## store the results

# COMMAND ----------

# rf_train_df.write.format('delta').mode('overwrite').option('overwriteSchema','True').option("path","dbfs:/mnt/client-002sap20p006-interbio/04_data_analysis/results/{}_RF_selected_top{}.delta".format(HC_subtype,K)).saveAsTable("{}_RF_selected_top{}".format(HC_subtype,K))

# COMMAND ----------

# MAGIC %md # load/filter/prepare data for training

# COMMAND ----------

rf_train_df=spark.read.table("{}_RF_selected_top{}".format(HC_subtype,K))
rf_selected_features=rf_train_df.select('feature_label').distinct().rdd.flatMap(lambda x: x).collect()

# COMMAND ----------

selected_df=spark.read.table('train_preprocess')
selected_df.limit(4).show()

# COMMAND ----------

topMarker_selected_df=selected_df.filter(selected_df['feature_label'].isin(rf_selected_features))
topMarker_selected_df.show(5)

# COMMAND ----------

long_test_df=spark.read.table('test_preprocess')
long_train_df=spark.read.table('train_preprocess')
long_test_df.limit(5).show()

# COMMAND ----------

pheno_col=[col for col in long_test_df.columns if 'intensity' not in col and 'feature_label' not in col]

# Step 2: Get the train/test datasets
test_filtered = long_test_df.filter(long_test_df['feature_label'].isin(rf_selected_features))
train_filtered = long_train_df.filter(long_train_df['feature_label'].isin(rf_selected_features))
wide_filtered_test_df = test_filtered.drop('intensity').groupBy(pheno_col).pivot('feature_label').agg({'intensity_fill_log_scaled':'max'})
wide_filtered_train_df = train_filtered.drop('intensity').groupBy(pheno_col).pivot('feature_label').agg({'intensity_fill_log_scaled':'max'})

mtb_col=[col for col in wide_filtered_test_df.columns if 'mtb' in col]
print(len(mtb_col))

# Step 2: Assemble features
assembler = VectorAssembler(inputCols=mtb_col, outputCol="features")
test_assembled = assembler.transform(wide_filtered_test_df)
train_assembled = assembler.transform(wide_filtered_train_df)


test_df = test_assembled.select('features',HC_subtype)
train_df = train_assembled.select('features',HC_subtype)
test_df=test_df.withColumnRenamed(HC_subtype,'label')
train_df=train_df.withColumnRenamed(HC_subtype,'label')

# COMMAND ----------

# MAGIC %md #logitstic training

# COMMAND ----------

lr = LogisticRegression(featuresCol='features', labelCol='label')

param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")

# COMMAND ----------

# stratified sampling for validation set
label_column = "label"
class_proportions = train_df.groupBy(label_column).count().collect()
class_proportions = {row[label_column]: row['count'] for row in class_proportions}

# COMMAND ----------

best_model = None
best_auc = 0.0
# Specify the column name containing the class labels
label_column = "label"

# Calculate class distribution proportions
class_proportions = train_df.groupBy(label_column).count().collect()
class_proportions = {row[label_column]: row['count'] for row in class_proportions}



# Perform the outer cross-validation
for rseed in range(5):
    # Split the DataFrame into train and test sets
    train_ratio = 0.8  # 80% for training, 20% for testing

    # Create training and validation datasets for the outer loop

    # Calculate the desired fractions for stratified split
    fractions = {
        label: 0.8  # 80% for training, adjust as needed
        for label, count in class_proportions.items()
    }

    # Perform the stratified test-train split
    train_data = train_df.sampleBy(label_column, fractions, seed=rseed*3)
    val_data = train_df.subtract(train_data)

    # Create an inner TrainValidationSplit
    innerCrossValidator = CrossValidator(estimator=lr,
                                    estimatorParamMaps=param_grid,
                                    evaluator=evaluator,
                                    numFolds=5)

    # Perform the inner cross-validation
    inner_model = innerCrossValidator.fit(train_data)

    # Get the best model from the inner cross-validation
    current_model = inner_model.bestModel

    # Evaluate the current model on the validation data
    curr_auc = evaluator.evaluate(current_model.transform(val_data))

    # Update the best model and best AUC if the current model performs better
    if curr_auc > best_auc:
        best_model = current_model
        best_auc = curr_auc

# Use the best model obtained from the outer cross-validation
final_model = best_model


# COMMAND ----------

pr_evaluator = BinaryClassificationEvaluator(metricName='areaUnderPR')
roc_evaluator = BinaryClassificationEvaluator(metricName='areaUnderROC')

# Evaluate the model on the testing data using AUC-ROC
predictions = final_model.transform(test_df)
pr_score = pr_evaluator.evaluate(predictions)
roc_score = roc_evaluator.evaluate(predictions)
probs_labels = predictions.select('probability', 'label').rdd.map(lambda row: (float(row['probability'][1]), float(row['label'])))

probs, labels = zip(*probs_labels.collect())
fpr, tpr, thresholds_roc = roc_curve(labels, probs)
precision, recall, thresholds_pr = precision_recall_curve(labels, probs)
roc_auc = auc(fpr, tpr)

auc_roc = roc_auc
pr_auc = auc(recall, precision)

# COMMAND ----------

sns.set(style='white')
plt.figure(figsize=(5, 3))
plt.plot(recall, precision, label='Precision-Recall curve (AUC = %0.2f)' % pr_auc)
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc_roc)

plt.legend(loc='lower left')
plt.title('Evaluation of {} prediction using top {} metabolites'.format(HC_subtype,K))
plt.show()

# COMMAND ----------

# Print the coefficients and intercept for logistic regression
print("Coefficients: " + str(best_model.coefficients))
print("Intercept: " + str(best_model.intercept))
print("ElasticNetParam: " + str(best_model.getElasticNetParam()))
print("RegParam: " + str(best_model.getRegParam()))

# COMMAND ----------


