# MLNetCourseWork
Data Science With C# And ML.NET Course Assignments

## Assignment 1: Loading Data
This assignment illustrates various data cleaning techniques including:
* Scaling / normalizing numeric units to lower (natural) ranges to help our model converge quickly to a solution
* One-hot encoding 
* Binning for geo-date

## Assignment 2: Heart Disease Binary Classifier
This assignment illustrates a binary classifier used to predict prevalence of heart disease symptoms.

## Assignment 3: House Price Prediction

## Assignment 4: Fraud Detection

## Assignment 5: Digit / Handwritten Character Recognition
This assignment involves taking the MNIST dataset and creating a model that can predict handwritten characters. 
![](digit_recognition_results.png)

## Assignment 6: Spam Detection
In this assignment, I use the *K-Fold Cross Validation* technique to ensure that the size of the dataset is sufficient for building out this model. The target for this determination is an average AUC (Area Under the Curve) of > 0.8. See below:
![](spam_detection_results.png)

On training and evaluating the model, we get the following results:

### Result Descriptions
* Accuracy: this is the number of correct predictions divided by the total number of predictions.
* AreaUnderRocCurve: a metric that indicates how accurate the model is: 0 = the model is wrong all the time, 0.5 = the model produces random output, 1 = the model is correct all the time. An AUC of 0.8 or higher is considered good.
* AreaUnderPrecisionRecallCurve: an alternate AUC metric that performs better for heavily imbalanced datasets with many more negative results than positive.
* F1Score: this is a metric that strikes a balance between Precision and Recall. It’s useful for imbalanced datasets with many more negative results than positive.
* LogLoss: this is a metric that expresses the size of the error in the predictions the model is making. A logloss of zero means every prediction is correct, and the loss value rises as the model makes more and more mistakes.
* LogLossReduction: this metric is also called the Reduction in Information Gain (RIG). It expresses the probability that the model’s predictions are better than random chance.
* PositivePrecision: also called ‘Precision’, this is the fraction of positive predictions that are correct. This is a good metric to use when the cost of a false positive prediction is high.
* PositiveRecall: also called ‘Recall’, this is the fraction of positive predictions out of all positive cases. This is a good metric to use when the cost of a false negative is high.
* NegativePrecision: this is the fraction of negative predictions that are correct.
* NegativeRecall: this is the fraction of negative predictions out of all negative cases.

## Case Study: Detecting Toxic Lingua
In this case study, I build a model that flags toxic comments (rude, disrespectful or likely to make someone leave) in a discussion. Using a dataset from Wikipedia's page comments downloaded from kaggle.com. 