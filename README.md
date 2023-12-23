This report is based on the wine quality data set available at the link:
https://archive.ics.uci.edu/ml/datasets/Wine+Quality.

The dataset contains physicochemical (inputs) and sensory (the output) variables for each observation. 
The aim of this report is to build a model in order to predict if a wine is good or not, in
other words in order to predict human wine taste preferences.

To achieve the aim of this report, the data set has been divided into three datasets: the train
data (about 60% of the units of the original dataset), the validation data (about 20% of the units
of the original dataset) and the test data (about 20% of the units of the original dataset).
After a preliminary univariate analysis, the dataset has been splitted into two groups: a wine
with a value of quality less or equal than 5 has been classied as bad wine (low quality), on
the contrary a wine with a value of quality more than 5 has been classied as good wine (high
quality). The relationship between each physiochemical property and quality, low or high, has
been investigated. Then, three different models have been tted using the training dataset: a
logistic regression model, a classication trees model and a neural networks model. Then, after a
comparison between the three models, the best one has been chosen and tested on the validation
dataset.

