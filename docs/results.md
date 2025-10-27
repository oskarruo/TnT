---
layout: page
title: Model Reports
---

## Logistic Regression

As the logistic regression model is implemented using statsmodels, it allows us to gather more detailed metrics about the model.
For example, using p-values with the significance level of 0.05 we can see that out of the features, **"balance"** and **"f0_std"** are statistically significant, with a p-value of less than or equal to 0.05. The coefficient of "balance" is also clearly the most significant. This implies that TED-speakers have lower balance-values, meaning that their speeches contain more pauses. The coefficient of "f0_std" is much less significant, but it does imply that TED-speakers tend to have more pitch variability.

<img src="images/logreg_coef_pval_graph.png" alt="Logistic Regression Coefficient p-value" height="350">

The confusion matrix of the model shows that the model does make a significant amount of wrong classifications, which tend to be equally distributed between the classes. The overall accuracy **0.6836 ± 0.0253** of the model is decent.

<img src="images/logreg_confusion_matrix.png" alt="Logistic Regression Confusion Matrix" height="350">

The ROC curve gives a slightly better perspective, with an AUC of **0.76 ± 0.03**, suggesting a good ability to distinguish between TED and non-TED talks.

<img src="images/logreg_roc_curve.png" alt="Logistic Regression ROC Curve" height="350">

Lastly, the Pseudo R-squared value of **0.1479 ± 0.0103** indicates a moderate explanatory power.

## Random Forest

Interestingly, the feature importance values of the random forest model are noticeably different to the logreg model's coefficients. Mainly, the importance of **"f0_std"** is clearly the largest, with **"balance"** coming second at a significantly lower value. The other two remain relatively insignificant. This indicates that, according to the random forest, pitch variability plays a dominant role in partitioning the data across the trees.

<img src="images/randomforest_feature_importance.png" alt="Random Forest Feature Importance" height="350">

The confusion matrix is quite similar to the logreg model, but there is a noticeable drop in overall accuracy, down to around **0.6488 ± 0.0198**.

<img src="images/randomforest_confusion_matrix.png" alt="Random Forest Confusion Matrix" height="350">

As with the logistic regression model, the ROC curve gives a slightly better rating, with an AUC of **0.72 ± 0.02**, also suggesting a good ability to distinguish between TED and non-TED talks.

<img src="images/randomforest_roc_curve.png" alt="Random Forest ROC Curve" height="350">

## Support Vector Machine (SVM)

Again the feature importance stays the same as in previous models. **"f0_std"** is again the highest at 0.66 and **"balance"** at the second place at 0.22. The two other features are staying at around 0.05.

<img src="images/svm_feature_importance.png" alt="SVM Feature Importance" height="350">

The overall accuracy is around **74%**. The confusion matrix has some differences because it is predicting more TED vs NON-TED talks than the previous models. Also there are differences in precision, recall and f1-score between 0 and 1 classifications but they are still looking alright.

<img src="images/svm_confusion_matrix.png" alt="SVM Confusion Matrix" height="350">

|              |   precision |   recall |   f1-score |    support |
|:-------------|------------:|---------:|-----------:|-----------:|
| 0            |    0.82     | 0.61     |   0.70     | 216        |
| 1            |    0.69     | 0.87     |   0.77     | 216        |
| accuracy     |             |          |   0.74     | 432        |
| macro avg    |    0.76     | 0.74     |   0.74     | 432        |
| weighted avg |    0.76     | 0.74     |   0.74     | 432        |

The ROC curve gives a slightly better perspective, with an AUC of **0.79**, suggesting a good ability to distinguish between TED and non-TED talks. This is close to 0.8 which is generally considered a good model.

<img src="images/svm_roc_curve.png" alt="SVM ROC Curve" height="350">

Using K-Fold, with five folds the SVM model achieved a mean accuracy of **71%** with a low standard deviation of **1.44%**. This indicates that the model's performance is stable and consistent with different data splits.

<img src="images/svm_kfold.png" alt="SVM k-folds cross-validation" height="350">

Below we can see how the **"f0_std"** really affects in the classification with SVM.

<img src="images/svm_classified.png" alt="SVM classification" height="350">

## K-nearest neighbors (KNN)

Again **"f0_std"** is the most important feature at 0.46 but now **"rate_of_speech"** is at second place at 0.27. **"balance"** is also up there at 0.2 and lastly **"articulation_rate"** at 0.065.

<img src="images/knn_feature_importance.png" alt="KNN Feature Importance" height="350">

The overall accuracy is around **70%**. Again the confusion matrix has some differences because it is predicting more TED vs NON-TED talks than the first 2 models, but is similar to SVM. Also there are differences in precision, recall and f1-score between 0 and 1 classifications but they are still looking alright.

<img src="images/knn_confusion_matrix.png" alt="KNN Confusion Matrix" height="350">

|              |   precision |   recall |   f1-score |    support |
|:-------------|------------:|---------:|-----------:|-----------:|
| 0            |    0.75     | 0.60     |   0.67     | 216        |
| 1            |    0.67     | 0.80     |   0.73     | 216        |
| accuracy     |             |          |   0.70     | 432        |
| macro avg    |    0.71     | 0.70     |   0.70     | 432        |
| weighted avg |    0.71     | 0.70     |   0.70     | 432        |

The ROC curve is quite similar with the SVM, with an AUC of **0.76**, suggesting a good ability to distinguish between TED and non-TED talks.

<img src="images/knn_roc_curve.png" alt="KNN ROC Curve" height="350">

Using K-Fold, with five folds the KNN model achieved a mean accuracy of **66.7%** with a standard deviation of **2.8%**. This indicates that the model's performance is not as good as in SVM for example, but can still make quite good predictions with different data splits.

<img src="images/knn_kfold.png" alt="KNN k-folds cross-validation" height="350">

## Comparison between model accuracies

In a graph showing a 95% confidence interval for the accuracies, the models are in a distinct order and can thus be ranked easily. **SVM is clearly the most accurate**, being the only model to achieve a mean accuracy of over 70%. It also has the lowest spread of the models.

<img src="images/model_accuracy_comparison.png" alt="Model Accuracy Comparison" height="350">

The next graph shows the AUC scores of the models displayed with their 95% confidence intervals, sorted from lowest mean AUC to highest. Again, **the leading model is SVM** with the mean AUC of 0.77. Logistic Regression model is almost as strong as SVM and has a lower spread which indicates more consistent performance.

<img src="images/model_auc_comparison.png" alt="Model AUC Comparison" height="350">

In conclusion, with the help of k-fold, we saw that SVM consistently outperforms the other models in terms of mean accuracy and mean AUC, while also showing relatively low variability across the folds. Logistic Regression is clearly the second best option and performs nearly as well as SVM. KNN and Random Forest have also achieved decent results but are still behind SVM and Logistic Regression in both accuracy and AUC. Overall, SVM would be our preferred choice for classifying TED and non-TED talks, with Logistic Regression as a strong alternative.
