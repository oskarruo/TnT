*[Back to the front page](./index.md/)*

# Methods

## Data collection

For TED-talks, we use a scraper utilizing TED's api and yt-dlp, which downloads the audios of each talk. We collected 4000 TED-audios, of which 1837 were stage talks. The TED-audios range from around 2.5 minutes to an hour.

For non-TED-talks, we use audios of youtube videos, specifically from public or hidden playlists. For getting them, we also use yt-dlp. We collected a total of 719 youtube audios. The audios range from around 1.5 minutes to an hour.

## Data preprocessing

We use the python library [myprosody](https://github.com/Shahabks/myprosody) to extract acoustic features of the audios. 

The features that are comparable between audios, which we therefore use in the modeling part, are:
- **"rate of speech"** (syllables/second using whole duration of the audio)
- **"articulation of speech"** (syllables/second using only speaking duration without pauses)
- **"balance"** (ratio of speaking duration without pauses / whole duration)
- **"f0_std"** (standard deviation of fundamental frequency distribution = pitch variability)

## Modeling

We use the following models for binary classification of the audios:

### Logistic regression

Logistic regression predicts the probability that an audio is a TED talk based on its acoustic features. Each feature has a coefficient that describes how strongly it pushes the prediction toward TED or non-TED. We use [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.formula.api.logit.html) for this because it allows us to extract more detailed metrics such as p-values.

### Random forest

Random forests are ensembles of decision trees. Each tree makes a prediction based on feature thresholds, and the forest averages these predictions. We use [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) for this.

### Support Vector Machine (SVM)

We are using SVM:s subclass SVC (Support Vector Classifier) to find an optimal decision boundary that separates the two classes. The model focuses on the points closest to the boundary which are called support vectors. We use [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) for this.

### K-Nearest Neighbors (KNN)

KNN classifies the talks based on the classes of its closest neighbors using the features. So basically it chooses if the talk is TED or non-TED by looking at the talks that seem similar to the one being classified. We use [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) for this.

## Training

We use K-fold cross-validation to split the dataset into 5 folds to get a reliable estimate of the models' accuracies. For random forests, we tune hyperparameters using grid search across n_estimators and max_depth.

## Evaluation

For evaluating the models we use:
- **Coefficient/Feature improtance**: Indicates which features contribute most to distinguishing TED from non-TED audios. For logistic regression, coefficients show direction and magnitude of influence; for others, importance scores show relative contribution.
- **Accuracy**: Proportion of correctly classified audios
- **AUC**: Measures the model’s ability to distinguish classes across thresholds
- **Confusion matrix**: Shows share of true positives, true negatives, false positives, and false negatives
- **Logistic regression p-values & pseudo-R²**: Indicate significance of each feature and model fit