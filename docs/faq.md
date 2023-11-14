# FAQ

## OrderedModel Training

### What are some challenges in training OrderedModel?

Challenges with OrderedModel training primarily revolve around overparameterization. Overparameterization can lead to models that are overly complex relative to the data, resulting in poor predictive performance on new, unseen data. Additionally, having too many parameters can also increase computational demands unnecessarily.

#### Example
In sprint 9, when the number of _tsfresh_ features (`tsfresh_num_features` in `parameters.yml`) was set to 20, the total feature count inflated to over 60. This significant increase in features caused OrderedModel to fail to run. The following error message was encountered:

```
24/10/2023 02:52 | kedro.pipeline.node | ERROR | Node 'model_train_train_model: train_model([ordered_model.model_specific_preprocessing_train,parameters,params:ordered_model]) -> [ordered_model.model_training_artefact]' failed with error: There should not be a constant in the model
```

As a temporary solution, we set the constants argument within the OrderedModel's fitting function to `False`:
~~~
model = OrderedModel(y_train,X_train,distr='probit', hasconst=False)
~~~
This action reduced the parameter space by not considering certain variables as constants, thereby alleviating the burden on the model's optimization algorithm. Refer [here](https://www.statsmodels.org/stable/examples/notebooks/generated/ordinal_regression.html#Using-formulas---no-constant-in-model) for more infomation. It is important to note that this was the only solution attempted due to time constraints. 

### How can one avoid overparameterization in OrderedModel training?

To mitigate overparameterization, consider the following strategies:
- **Feature Selection**: Select only the most relevant features for training your model.
- **Cross-Validation**: Employ cross-validation techniques to evaluate the model's ability to generalize to unseen data.

### Can the process of feature selection be automated?

Yes, the feature selection process can be automated. Python libraries such as [tsfresh](https://tsfresh.readthedocs.io/en/latest/) and [scikit-learn](https://scikit-learn.org/) provide  methods for automatically selecting the best features based on performance metrics.

### What are _tsfresh_ features and their role in OrderedModel training?

In the context of time series, _tsfresh_ is often used to generate a vast number of features calculated from the given time series data. For detailed information on the features supported by _tsfresh_, refer to the [tsfresh documentation](https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html). These features are used to capture the underlying patterns of the time series for use in various ML models, including OrderedModel.

### What are some tips on tuning the number of _tsfresh_ features?

When adjusting the number of _tsfresh_ features:
- Start with the default settings and refine based on the model's initial performance.
- Incrementally include more features while monitoring performance.
- Set a performance target to guide when to stop adding features.
- Employ domain knowledge to keep only features pertinent to the specific problem area.

For an integrated approach to _tsfresh_ feature adjustment for this project, review the section on Time-Dependent Feature Engineering within the [Data Pipeline Documentation](./data-pipeline#6-time-dependent-feature-engineering).


## Adding in a New Model


## Automated Hyperparameter Search

### What is hyperparameter search and why is it important?
Hyperparameter search is the practice of selecting the ideal combination of hyperparameters that govern the learning process of a machine learning algorithm. These settings, unlike model parameters, are not derived from the training data but are set prior to the learning process and directly influence model performance.

Optimal hyperparameter settings are crucial as they can significantly enhance model accuracy and efficiency, leading to superior outcomes for the given machine learning challenge.

### Can hyperparameter search be automated?
Indeed, the hyperparameter search can be automated. Tools such as Optuna, and Scikit-learn's GridSearchCV and RandomizedSearchCV, provide sophisticated methods for automating this process, streamlining model optimisation. 

### Which hyperparameter search technique should I use?
The choice of technique often depends on the complexity of the model and the computational resources available. 

| Technique | Ideal Usage Scenario | Description |
| :- | - | - |
| [GridSearchCV (Scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)       | Optimal for models with fewer hyperparameters where exhaustive search is viable, prioritising precision over speed.                             | Systematically probes every combination within a predefined hyperparameter grid, ensuring the identification of the best parameters within the grid's boundaries. |
| [RandomizedSearchCV (Scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) | Suited to larger hyperparameter domains where exhaustive search is impractical. Good for a rapid overview with limited computational resources. | Selects hyperparameter sets at random from a defined distribution. This can lead to faster but potentially less precise results compared to GridSearchCV.         |
| [Optuna](https://optuna.readthedocs.io/en/stable/)                            | Recommended for complex and high-dimensional spaces, with the advantage of successive trials informing subsequent search paths.                 | Employs Bayesian optimization to traverse the hyperparameter space efficiently, concentrating on areas indicated as promising by past trial outcomes.             |

### How to evaluate the effectiveness of a hyperparameter search?
The effectiveness is typically evaluated by the improvement in the model's performance on a validation set. Consider to monitor the performance metrics relevant to your specific problem, such as: 
- Accuracy, precision, recall or F1 score for classification tasks; or
- Mean squared error (MSE), root mean square error (RMSE) or mean absolute error (MAE) for regression tasks.

Additionally, consider the time and computational resources used by the search process.

### How do I determine the range of values to explore in a hyperparameter search?
Determining the value range for hyperparameters can be guided by past empirical evidence, expertise in the relevant domain, or by initiating with a broad spectrum and iteratively refining this based on initial search findings.
