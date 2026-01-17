import pandas as pd
import numpy as np

attack_path = "/kaggle/input/cic-ddos2019-30gb-full-dataset-csv-files/01-12/DrDoS_DNS.csv"
benign_path = "/kaggle/input/cic-ddos2019-30gb-full-dataset-csv-files/03-11/UDP.csv"

df_attack = pd.read_csv(attack_path, low_memory=False)
df_benign = pd.read_csv(benign_path, low_memory=False)

df_attack = df_attack.sample(50000, random_state=42)
df_benign = df_benign.sample(50000, random_state=42)

df_attack["label"] = 1
df_benign["label"] = 0

df_cic = pd.concat([df_attack, df_benign], ignore_index=True)

cic_drop = [
    "Flow ID", " Source IP", " Destination IP",
    " Timestamp", " Label", "Unnamed: 0"
]

df_cic_ml = df_cic.drop(columns=cic_drop)

df_cic_ml = df_cic_ml.select_dtypes(include=["number"])

df_cic_ml.replace([np.inf, -np.inf], np.nan, inplace=True)
df_cic_ml.dropna(inplace=True)

X_cic = df_cic_ml.drop(columns=["label"])
y_cic = df_cic_ml["label"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cic,
    y_cic,
    test_size=0.3,
    random_state=42,
    stratify=y_cic
)

pipe_lr_cic = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

pipe_lr_cic.fit(X_train_c, y_train_c)


y_pred_lr_c = pipe_lr_cic.predict(X_test_c)

print(classification_report(y_test_c, y_pred_lr_c))
confusion_matrix(y_test_c, y_pred_lr_c)

rf_cic = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_cic.fit(X_train_c, y_train_c)

# Κρατάμε attack categories
df_unsw_mc = df_unsw.copy()

# Αφαιρούμε μη χρήσιμες στήλες
drop_cols = ["proto", "service", "state", "label"]
df_unsw_mc = df_unsw_mc.drop(columns=drop_cols)

# Target = attack category
y = df_unsw_mc["attack_cat"]
X = df_unsw_mc.drop(columns=["attack_cat"])

# Κρατάμε κατηγορίες με >= 1000 δείγματα
valid_classes = y.value_counts()
valid_classes = valid_classes[valid_classes >= 1000].index

mask = y.isin(valid_classes)
X = X[mask]
y = y[mask]

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_enc = le.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc,
    test_size=0.3,
    random_state=42,
    stratify=y_enc
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=2000,
        multi_class="multinomial",
        n_jobs=-1
    ))
])

pipe_lr.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

y_pred = pipe_lr.predict(X_test)

print(classification_report(
    y_test,
    y_pred,
    target_names=le.classes_
))

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print(classification_report(
    y_test,
    y_pred_rf,
    target_names=le.classes_
))