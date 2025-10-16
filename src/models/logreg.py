import glob
import pandas as pd
import re
import os
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold


# logistic regression model class
class LogReg:
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

        columns_to_use = ["rate_of_speech", "articulation_rate", "balance", "f0_std"]
        self.x_playlists = playlists[columns_to_use]
        self.x_teds = teds[columns_to_use]

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

        x_train_const = sm.add_constant(x_train)
        x_test_const = sm.add_constant(x_test)

        self.model = sm.Logit(y_train, x_train_const).fit(disp=False)
        self.y_pred_prob = self.model.predict(x_test_const)
        self.y_pred = (self.y_pred_prob >= 0.5).astype(int)

    def print_summary(self):
        print(self.model.summary())

    def get_summary_df(self):
        summary_df = pd.concat(
            [
                self.model.params.rename("coef"),
                self.model.pvalues.rename("p_value"),
                self.model.bse.rename("std_err"),
                self.model.conf_int().rename(columns={0: "ci_lower", 1: "ci_upper"}),
                pd.Series(
                    self.model.llf, index=self.model.params.index, name="log_likelihood"
                ),
                pd.Series(self.model.aic, index=self.model.params.index, name="AIC"),
                pd.Series(self.model.bic, index=self.model.params.index, name="BIC"),
                pd.Series(
                    self.model.prsquared,
                    index=self.model.params.index,
                    name="pseudo_R2",
                ),
            ],
            axis=1,
        )
        return summary_df

    def print_metrics(self):
        print(classification_report(self.y_test, self.y_pred))

    def get_metrics(self):
        return classification_report(self.y_test, self.y_pred, output_dict=True)

    def predict(self, x):
        proba = self.predict(x)
        return (proba >= 0.5).astype(int)

    def predict_proba(self, x):
        x_const = sm.add_constant(x, has_constant="add")
        return self.model.predict(x_const)

    # this function is for creating a dictionary of cross-validated results
    def cross_validate(self, k=5):
        x = pd.concat([self.x_playlists, self.x_teds], ignore_index=True)
        y = np.array([0] * len(self.x_playlists) + [1] * len(self.x_teds))

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.random_state)

        accs = []
        aucs = []
        coefs = []
        pvals = []
        r2s = []

        mean_fpr = np.linspace(0, 1, 100)
        tprs = []

        y_true_all = []
        y_pred_all = []

        for train_idx, test_idx in skf.split(x, y):
            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            x_train_const = sm.add_constant(x_train)
            x_test_const = sm.add_constant(x_test)

            model = sm.Logit(y_train, x_train_const).fit(disp=False)
            y_pred_prob = model.predict(x_test_const)
            y_pred = (y_pred_prob >= 0.5).astype(int)

            acc = np.mean(y_pred == y_test)
            auc = roc_auc_score(y_test, y_pred_prob)
            accs.append(acc)
            aucs.append(auc)
            r2s.append(model.prsquared)

            coefs.append(model.params)
            pvals.append(model.pvalues)

            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            tprs.append(np.interp(mean_fpr, fpr, tpr))

            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)

        self.cv_results = {
            "accuracy_mean": np.mean(accs),
            "accuracy_std": np.std(accs),
            "auc_mean": np.mean(aucs),
            "auc_std": np.std(aucs),
            "r2_mean": np.mean(r2s),
            "r2_std": np.std(r2s),
            "coef_mean": pd.DataFrame(coefs).mean(),
            "coef_std": pd.DataFrame(coefs).std(),
            "pval_mean": pd.DataFrame(pvals).mean(),
            "pvals": pd.DataFrame(pvals),
            "pval_std": pd.DataFrame(pvals).std(),
            "mean_fpr": mean_fpr,
            "mean_tpr": np.mean(tprs, axis=0),
            "std_tpr": np.std(tprs, axis=0),
            "y_true_all": np.array(y_true_all),
            "y_pred_all": np.array(y_pred_all),
        }

        return self.cv_results
