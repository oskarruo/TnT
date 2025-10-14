import glob
import pandas as pd
import re
import os
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class LogReg:
    def __init__(self):
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
        x_playlists = playlists[columns_to_use]
        x_teds = teds[columns_to_use]

        if len(x_playlists) > len(x_teds):
            x_playlists = x_playlists.sample(n=len(x_teds), random_state=1)
        elif len(x_teds) > len(x_playlists):
            x_teds = x_teds.sample(n=len(x_playlists), random_state=1)

        x = pd.concat([x_playlists, x_teds], ignore_index=True)
        y = [0] * len(x_playlists) + [1] * len(x_teds)

        x_train, x_test, y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, random_state=1, stratify=y
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
