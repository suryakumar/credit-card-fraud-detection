# credit-card-fraud-detection
Credit Card Fraud detection models

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The repository consists to 2 notebooks viz. 

__*data_exploration*__: This consists of basic data exploration which includes 
 - Exploring the various datatypes across the columns
 - Understanding the distribution of fraud transactions vs valid transactions
 - Looking at a heatmap of the correlation matrices across the vectors
 

__*NN_with_undersampling*__: 

*Task*: Develop a neural network model to predict fraud transactions.

*Approach*: As seen during data exploration, the underlying dataset is extremely skewed (Only ~500 fraud transactions exist in ~75k total transactions). Hence, a standard neural network (or any other ensemble based classifier) will not work.
As a first step, the dataset is scaled between 0 & 1. To be more specific, the Amount and Time column values are scaled using a simple MinMaxScaler.

Now, we aim to train a neural network model to classify a given transaction in this dataset as 'fraud' or 'valid'. Within the dataset, the *Class* variable represents this binary aspect with 1 being fraud and 0 being a valid transaction.

The core challenge with training a neural network model is the imbalanced dataset with very few fraud transacitons, is that a model which predicts all transactions as valid will have a very low loss (by the virtue of the imbalance).

We address this problem by performing imbalanced sampling i.e. for each batch within training loop we sample data with almost equal number of fraud and valid transactions. Furthermore, for a batch within an epoch, we sample fraud transactions with resampling being true while valid transactions only once. The loss criterion also gives higher weightage to a fraud transaction being classified incorrectly via weights in *BCELogitsLoss*

One more nuance, when splitting the data into training and test sets, we stratify the sampling based on the target variable (to ensure that the ratio of valid:fraud transactions remains similar in both training and test sets).

The notebook shows the final training metrics as well as the confusion matrix which showcases that the model peforms well on the test dataset.
