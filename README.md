# [TnT: TED or non-TED](https://oskarruo.github.io/TnT/)

*A data science project that tries to answer the question "How to Speak Like a TED Talk-er"*

## Documentation

Documentation of the project can be found on [https://oskarruo.github.io/TnT/](https://oskarruo.github.io/TnT/) or in the [docs](./docs) directory.

## Data, Models & Running the experiments

To scrape more data and/or use the models and/or generate plots:

1. Ensure that [FFmpeg](https://www.ffmpeg.org/) is installed and in PATH
2. Install poetry from `https://python-poetry.org/`
3. Clone the project
```
git clone https://github.com/oskarruo/TnT.git
cd TnT
```
4. Install dependencies with `poetry install`

### Data

The data used in the project, located in the [data/csv](./data/csv) directory, is free to use.
The directory contains data of prosodic features of TED-talks and YouTube playlists.
More data can be scraped by running the scripts in the [src](./src) directory.

After the previously mentioned dependencies have been installed, more TED data can be scraped with:
```
cd src
python ted_scrape_and_analyze.py n n_per_time sorting
```
where:
- n: int = amount of speeches to download
- n_per_time: int = amount of speeches to download and analyze at once
- sorting: string = sort by "popular" or "newest" speeches

More YouTube data can be scraped with
```
cd src
python playlist_scrape_and_analyze.py url n_per_time
```
where:
- url: str = playlist url to analyze
- n_per_time: int = amount of vids to download and analyze at once

**YouTube videos longer than 1 hour will be ignored by the scraper**

### Models

The logistic regression and random forest models can be initialized followingly:
```
from models.logreg import LogReg
from models.randomforest import RandomForest

logreg_model = LogReg()
rf_model = RandomForest()
```
This will train the models with all of the data in the data/csv directory.

#### Methods

**predict**
```
# logreg_model.predict()
# rf_model.predict()

# Example:
data = pd.DataFrame([[4, 5, 0.7, 30.4]])
logreg_model.predict(data) -> 0
```
This will return the predicted class (**1 for TED, 0 for non-TED**).
The input must be a pandas DataFrame with the "rate_of_speech, articulation_rate, balance, f0_std" values of the audio.

**predict_proba**
```
# logreg_model.predict_proba()
# rf_model.predict_proba()

# Example:
data = pd.DataFrame([[4, 5, 0.7, 30.4]])
rf_model.predict_proba(data) -> 0.03
```
This will return the probability of class 1 (TED).
The input must be a pandas DataFrame with the "rate_of_speech, articulation_rate, balance, f0_std" values of the audio.

**print_metrics**
```
logreg_model.print_metrics()
rf_model.print_metrics()
```
This will print precision, recall, f1-score and accuracy.

**print_summary**
```
logreg_model.print_summary()
```
This will print the statsmodels summary for the logistic regression model.

### Generating plots

To generate plots for the logistic regression and random forest models, run:
```
cd src
python generate_plots_and_stats.py
```
This will generate plots of the two models to the [docs/images](./docs/images) directory & print accuracies and the pseudo R-squared for logistic regression.
