from sklearn.model_selection import train_test_split
import pandas as pd

train_path = "/kaggle/input/unswnb15/UNSW_NB15_training-set.parquet"
test_path  = "/kaggle/input/unswnb15/UNSW_NB15_testing-set.parquet"

df_unsw = pd.concat(
    [
        pd.read_parquet(train_path),
        pd.read_parquet(test_path)
    ],
    ignore_index=True
)


unsw_drop = ["proto", "service", "state", "attack_cat"]
df_unsw_ml = df_unsw.drop(columns=unsw_drop)

X_unsw = df_unsw_ml.drop(columns=["label"])
y_unsw = df_unsw_ml["label"]

X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(
    X_unsw,
    y_unsw,
    test_size=0.3,
    random_state=42,
    stratify=y_unsw
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe_lr_unsw = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

pipe_lr_unsw.fit(X_train_u, y_train_u)

from sklearn.metrics import classification_report, confusion_matrix

y_pred_lr_u = pipe_lr_unsw.predict(X_test_u)

print(classification_report(y_test_u, y_pred_lr_u))
confusion_matrix(y_test_u, y_pred_lr_u)


from sklearn.ensemble import RandomForestClassifier

rf_unsw = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_unsw.fit(X_train_u, y_train_u)

