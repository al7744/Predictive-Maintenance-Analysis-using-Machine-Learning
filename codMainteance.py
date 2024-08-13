import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
import warnings

warnings.filterwarnings('ignore')

file_path = 'predictive_maintenance.csv'
df = pd.read_csv(file_path)
print(df.head())
print(df.info())
print(df.isna().sum())
print(df.duplicated().sum())
df.drop(df.columns[:2], axis=1, inplace=True)

cat_cols = df.select_dtypes(include='O').columns.tolist()
print(cat_cols)
for col in cat_cols:
    print(f"Value counts for column '{col}':")
    print(df[col].value_counts())
    print()

df_f = df[df['Target'] == 1]
print(df_f['Failure Type'].value_counts())
idx = df_f[df_f['Failure Type'] == 'No Failure'].index
df.drop(idx, axis=0, inplace=True)
df_f = df[df['Target'] == 0]
print(df_f['Failure Type'].value_counts())
idx = df_f[df_f['Failure Type'] == 'Random Failures'].index
df.drop(idx, axis=0, inplace=True)
print(df.info())

num_cols = df.select_dtypes(exclude='O').columns.tolist()
print(num_cols)

df.replace([np.inf, -np.inf], np.nan, inplace=True)

plt.figure(figsize=(15, 10))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data=df, x=col)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

sns.pairplot(df, hue='Target')
plt.show()

x = df.drop(['Target', 'Failure Type'], axis=1)
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

cat_cols = X_train.select_dtypes(include='O').columns.tolist()
num_cols = X_train.select_dtypes(exclude='O').columns.tolist()

enc = OneHotEncoder(handle_unknown='ignore')
sc = StandardScaler()
ct = ColumnTransformer(
    [
        ('encoding', enc, cat_cols),
        ('scaling', sc, num_cols)
    ]
)

x_train = ct.fit_transform(X_train)
x_test = ct.transform(X_test)

clf = LogisticRegression()
clf.fit(x_train, y_train)
logistic_score = clf.score(x_test, y_test)
print(f'Logistic Regression Test Score: {logistic_score:.4f}')

svc = SVC()
svc.fit(x_train, y_train)
svc_score = svc.score(x_test, y_test)
print(f'SVC Score: {svc_score:.4f}')

rsvc = SVC(kernel='rbf')
rsvc.fit(x_train, y_train)
rsvc_score = rsvc.score(x_test, y_test)
print(f'RBF SVC Score: {rsvc_score:.4f}')

rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(x_train, y_train)
clf.fit(X_resampled, y_resampled)
undersampling_score = clf.score(x_test, y_test)
print(f'Logistic Regression Score after undersampling: {undersampling_score:.4f}')

X_resampled, y_resampled = SMOTE().fit_resample(x_train, y_train)
clf.fit(X_resampled, y_resampled)
smote_score = clf.score(x_test, y_test)
print(f'Logistic Regression Score after SMOTE: {smote_score:.4f}')

over = SMOTE(sampling_strategy=0.5)
und = RandomUnderSampler()
pipe = Pipeline(
    [
        ('o', over),
        ('u', und)
    ]
)
x_s, y_s = pipe.fit_resample(x_train, y_train)
clf.fit(x_s, y_s)
combined_score = clf.score(x_test, y_test)
print(f'Logistic Regression Score after over and under-sampling: {combined_score:.4f}')

print(f"Logistic Regression Score: {logistic_score:.4f}")
print(f"SVC Score: {svc_score:.4f}")
print(f"RBF SVC Score: {rsvc_score:.4f}")
print(f"Logistic Regression Score after undersampling: {undersampling_score:.4f}")
print(f"Logistic Regression Score after SMOTE: {smote_score:.4f}")
print(f"Logistic Regression Score after over and under-sampling: {combined_score:.4f}")

numeric_df = df.select_dtypes(include=[np.number])
numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)

plt.figure(figsize=(12, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

plt.figure(figsize=(15, 10))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(4, 3, i)
    sns.countplot(data=df, x=col)
    plt.title(f'Count Plot of {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 3, i)
    sns.regplot(x=df[col], y=df['Target'])
    plt.title(f'Regression Plot of {col} vs Target')
plt.tight_layout()
plt.show()

joblib.dump(clf, 'logistic_regression_model.joblib')
loaded_model = joblib.load('logistic_regression_model.joblib')
test_score = loaded_model.score(x_test, y_test)
print(f'Loaded Model Test Score: {test_score:.4f}')
