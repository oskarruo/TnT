import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from pathlib import Path

base_path = Path(__file__).resolve().parent.parent.parent
ted_df = pd.read_csv(base_path / "data/csv/analyzed_speeches_4000_popular.csv")
ted_df = ted_df[~ted_df['type_name'].str.startswith(
    ('TED-Ed', 'Original', 'Best of', 'TED Salon', 'TED Institute'))]
ted_df = ted_df.reset_index(drop=True)
ted_df = ted_df.dropna(subset=['rate_of_speech', 'articulation_rate', 'balance',
                               'original_duration', 'speaking_duration', 'f0_std'])
ted_df['is_ted'] = 1
ted_df['Minutes'] = ted_df['original_duration'] / 60
ted_df['Pauses_duration'] = ted_df['original_duration'] - ted_df['speaking_duration']
ted_df['Pauses_per_minute'] = ted_df['Pauses_duration'] / ted_df['Minutes']

def prepare_non_ted(path):
    df = pd.read_csv(base_path / path)
    df = df.dropna(subset=['rate_of_speech', 'articulation_rate', 'balance',
                           'original_duration', 'speaking_duration', 'f0_std'])
    df['is_ted'] = 0
    df['Minutes'] = df['original_duration'] / 60
    df['Pauses_duration'] = df['original_duration'] - df['speaking_duration']
    df['Pauses_per_minute'] = df['Pauses_duration'] / df['Minutes']
    return df

non_ted_files = [
    "data/csv/analyzed_playlist_PL_4c34HZDoN6Ysc_Xw1V3V-M9KESB9bJ9.csv",
    "data/csv/analyzed_playlist_PLFf_-1kTMSNH8k_G-51W4w09bgl98AR1x.csv",
    "data/csv/analyzed_playlist_PLEL2J-7Brhes50MA_1VmohNe4v33UH6Df.csv",
    "data/csv/analyzed_playlist_PLgVhcWtOHMTq7_y2VAsfv3sJHDTb8XP9q.csv",
    "data/csv/analyzed_playlist_PL_K7XH1AIG8wZtQSM56Tyc-CR9ypvCbrF.csv",
    "data/csv/analyzed_playlist_PLnwMNodmyz8VCS4nXtd4qu7wP0U0b9y3I.csv"
]

non_ted_dfs = [prepare_non_ted(f) for f in non_ted_files]
non_ted_all = pd.concat(non_ted_dfs, ignore_index=True)

ted_train, ted_test = train_test_split(ted_df, test_size=0.3, random_state=42)
non_ted_train, non_ted_test = train_test_split(non_ted_all, test_size=0.3, random_state=42)

min_train = min(len(ted_train), len(non_ted_train))
min_test = min(len(ted_test), len(non_ted_test))

ted_train = ted_train.sample(n=min_train, random_state=42)
non_ted_train = non_ted_train.sample(n=min_train, random_state=42)
ted_test = ted_test.sample(n=min_test, random_state=42)
non_ted_test = non_ted_test.sample(n=min_test, random_state=42)

train_df = pd.concat([ted_train, non_ted_train], ignore_index=True)
test_df = pd.concat([ted_test, non_ted_test], ignore_index=True)

print(f"\n🔹 Train data: TED={len(ted_train)}, NON-TED={len(non_ted_train)}")
print(f"🔹 Test data: TED={len(ted_test)}, NON-TED={len(non_ted_test)}")

features = ['rate_of_speech', 'articulation_rate', 'balance', 'f0_std']
X_train = train_df[features]
y_train = train_df['is_ted']
X_test = test_df[features]
y_test = test_df['is_ted']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
print("\n✅ KNN-model trained successfully!")

y_pred = knn_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\n✅ KNN accuracy: {accuracy:.2f}%.\n")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, zero_division=0))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Feature Importances ---")
perm_importance = permutation_importance(knn_model, X_test_scaled, y_test,
                                         n_repeats=10, random_state=42)

importances = perm_importance.importances_mean
percent_importances = 100 * importances / importances.sum()

for i, feat in enumerate(features):
    print(f"{feat}: {percent_importances[i]:.2f}%")