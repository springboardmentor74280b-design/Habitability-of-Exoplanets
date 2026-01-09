import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("clean_exoplanet.csv.csv")
df = df.dropna(subset=['radius', 'mass', 'orbital_period', 'star_teff'])


df['pl_rade'] = df['radius']
df['pl_bmasse'] = df['mass']
df['pl_orbper'] = df['orbital_period']
df['pl_eqt'] = df['star_teff'] 
features = ['pl_rade', 'pl_bmasse', 'pl_orbper', 'pl_eqt']
df_features = df[features]

df_features['habitable'] = df_features['pl_eqt'].apply(lambda x: 1 if 180 <= x <= 310 else 0)

X = df_features.drop('habitable', axis=1)
y = df_features['habitable']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

from sklearn.metrics import accuracy_score, precision_score, recall_score
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
