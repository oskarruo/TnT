import glob
import pandas as pd
import re
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, roc_curve


# random forest model class
class RandomForest:
    def __init__(self, random_state=1):
        self.random_state = random_state
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "..", "..", "data", "csv", "*.csv")
        csv_files = glob.glob(csv_path)
        playlists = pd.DataFrame()
        teds = pd.DataFrame()

        for file in csv_files:
            if re.match(r"^analyzed_playlist", os.path.basename(file)):
                playlists = pd.concat([playlists, pd.read_csv(file)], ignore_index=True)
            elif re.match(r"^analyzed_speeches", os.path.basename(file)):
                teds = pd.concat([teds, pd.read_csv(file)], ignore_index=True)

        playlists = playlists.drop_duplicates(subset=["url"])
        teds = teds.drop_duplicates(subset=["slug"])
        teds = teds[teds["type_id"] == 1]

        self.features = ["rate_of_speech", "articulation_rate", "balance", "f0_std"]
        self.x_playlists = playlists[self.features]
        self.x_teds = teds[self.features]

        if len(self.x_playlists) > len(self.x_teds):
            self.x_playlists = self.x_playlists.sample(
                n=len(self.x_teds), random_state=self.random_state
            )
        elif len(self.x_teds) > len(self.x_playlists):
            self.x_teds = self.x_teds.sample(
                n=len(self.x_playlists), random_state=self.random_state
            )

        x = pd.concat([self.x_playlists, self.x_teds], ignore_index=True)
        y = [0] * len(self.x_playlists) + [1] * len(self.x_teds)

        x_train, x_test, y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        param_grid = {
            "n_estimators": [100, 300, 500],
            "max_depth": [None, 30, 50],
        }

        grid = GridSearchCV(
            RandomForestClassifier(random_state=self.random_state), param_grid
        )
        grid.fit(x_train, y_train)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        self.y_pred = self.model.predict(x_test)
        self.y_pred_prob = self.model.predict_proba(x_test)[:, 1]

    def print_metrics(self):
        print(classification_report(self.y_test, self.y_pred))

    def get_metrics(self):
        return classification_report(self.y_test, self.y_pred, output_dict=True)

    def predict_proba(self, x):
        return pd.Series(self.model.predict_proba(x)[:, 1][0])

    def predict(self, x):
        proba = self.predict_proba(x)
        return (proba >= 0.5).astype(int)

    # this function is for creating a dictionary of cross-validated results
    def cross_validate(self, k=5):
        x = pd.concat([self.x_playlists, self.x_teds], ignore_index=True)
        y = np.array([0] * len(self.x_playlists) + [1] * len(self.x_teds))

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.random_state)

        accs = []
        aucs = []
        importances = []

        mean_fpr = np.linspace(0, 1, 100)
        tprs = []

        y_true_all = []
        y_pred_all = []

        for train_idx, test_idx in skf.split(x, y):
            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = RandomForestClassifier(
                random_state=self.random_state, **self.best_params
            )
            model.fit(x_train, y_train)
            y_pred_prob = model.predict_proba(x_test)[:, 1]
            y_pred = (y_pred_prob >= 0.5).astype(int)

            acc = np.mean(y_pred == y_test)
            auc = roc_auc_score(y_test, y_pred_prob)
            accs.append(acc)
            aucs.append(auc)

            importances.append(model.feature_importances_)

            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            tprs.append(np.interp(mean_fpr, fpr, tpr))

            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)

        self.cv_results = {
            "accuracy_mean": np.mean(accs),
            "accuracy_std": np.std(accs),
            "auc_mean": np.mean(aucs),
            "auc_std": np.std(aucs),
            "imp_mean": np.mean(importances, axis=0),
            "imp_std": np.std(importances, axis=0),
            "mean_fpr": mean_fpr,
            "mean_tpr": np.mean(tprs, axis=0),
            "std_tpr": np.std(tprs, axis=0),
            "y_true_all": np.array(y_true_all),
            "y_pred_all": np.array(y_pred_all),
        }

        return self.cv_results
