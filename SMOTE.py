import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

df = pd.read_csv(r"C:\Users\Menaka\OneDrive\Desktop\INFOSYS PROJECT DOC\cleaned_exoplanet_dataset.csv")
target_col = "P_HABITABLE"

X = df.drop(columns=[target_col])
y = df[target_col]

#SMOTe work on numerical data so i'm performing the encoding here
cat_cols = X.select_dtypes(include=['object']).columns

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

#we shouldn't apply smote bef splitting ( Never apply SMOTE before splitting → data leakage! )
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

#before applying smote technique
print("Before SMOTE:")
print(y_train.value_counts())

#here i'm applying smote(SMOTE creates synthetic habitable samples)
smote = SMOTE(
    sampling_strategy='auto',
    k_neighbors=5,
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

#after applying smote technique
print("After SMOTE:")
print(y_train_smote.value_counts())

#X_train_smote, y_train_smote → training  (X_test, y_test → testing )
print(X_train_smote.shape)
print(X_test.shape)

################          #####################

###### To add Smote flagged column in the daatset####
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

n_original = len(X_train)
n_total = len(X_train_smote)
n_synthetic = n_total - n_original

smote_flag = (
    ["original"] * n_original +
    ["synthetic"] * n_synthetic
)

X_train_smote = X_train_smote.copy()
X_train_smote["smote_flag"] = smote_flag

y_train_smote = pd.DataFrame(y_train_smote, columns=["habitable"])

#Combined features and target var
train_smote_full = pd.concat(
    [X_train_smote, y_train_smote],
    axis=1
)
#it will give the count
train_smote_full["smote_flag"].value_counts()

train_smote_full.to_csv(
    "train_dataset_with_smote_flag.csv",
    index=False
)

################     ##################


######i want a smote applied dataset separately so i used thiss #####
X_train_smote.to_csv("X_train_smote.csv", index=False)
y_train_smote.to_csv("y_train_smote.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)




