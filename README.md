Elastic Net Regression with Grid Search


This program uses elastic net regression with scikit-learn to create a model to predict house prices based on a variety of features. The data used can be found in the repository.
Elastic net regression works like normal regression except that it uses a combination of L1 and L2 regularisation techniques to punish coefficients that are too high. in scikit-learn you can adjust ElasticNet() with parameters that modify the proportion of L1 and L2 techniques and the overall weight given to regularisation.

The grid search capability in scikit-learn allows you to test multiple parameter values in one go and it will give you the best pair which you can then use to predict on test data.
