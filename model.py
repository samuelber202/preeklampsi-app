# %% [markdown]
# # Severe Preeclampsia Classification

# %%
# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ## Data Profiling

# %%
# Importing Dataset
df = pd.read_csv('data/data.csv')

# %%
# Previewing Dataset
df.head()

# %%
# Dataset Information
df.info()

# %%
# Dataset Description
df.describe()

# %%
# Checking Missing Values
df.isna().sum()

# %%
# Show Missing Values
df[df.isna().any(axis=1)]

# %%
# Plot with pie chart
df['Prognosis'].value_counts().plot(kind='pie', autopct='%1.1f%%', shadow=True)
plt.title('Prognosis')
plt.show()

# %% [markdown]
# ## Data Cleaning

# %%
# Remove ID Column
df.drop('id', axis=1, inplace=True)

# %%
# Convert Weight from String to Float
df['Weight'] = df['Weight'].str.replace(',', '.').astype(float)

# Replace Null Values in Weight with the Mean
df['Weight'] = df['Weight'].fillna(df['Weight'].mean())

# %%
# Remove the remaining rows with missing values
df.dropna(inplace=True)

# %%
# Check if there are any remaining missing values
df.isna().sum()

# %% [markdown]
# ## Data Transformation

# %%
# Convert Creatinine, Hemoglobin, Leukocytes, Hematocrit, Erythrocytes from String to Float
df['Creatinine'] = df['Creatinine'].str.replace(',', '.').astype(float)
df['Hemoglobin'] = df['Hemoglobin'].str.replace(',', '.').astype(float)
df['Leukocytes'] = df['Leukocytes'].str.replace(',', '.').astype(float)
df['Hematocrit'] = df['Hematocrit'].str.replace(',', '.').astype(float)
df['Erythrocytes'] = df['Erythrocytes'].str.replace(',', '.').astype(float)

# %%
# Dataset datatypes
df.dtypes

# %%
# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Proteinuria'] = le.fit_transform(df['Proteinuria'])
df['Delivery Method'] = le.fit_transform(df['Delivery Method'])
df['Prognosis'] = le.fit_transform(df['Prognosis'])

# %%
df.dtypes

# %%
# Delivery Method : Nullipara = 0, SC = 1, Spontaneous = 2
# Prognosis : Moderate Preeclamsia = 0, Severe Preeclampsia = 1
# Proteinuria : 0 = 0, 1+ = 1, 2+ = 2, 3+ = 3, 4+ = 4 
df.head()

# %%
# Check if there are any outliers in the dataset with boxplots
df[['Age', 'Height', 'Weight', 'Gestational Age', 'Gravida', 'Parity']].boxplot()
# Change plot size in inches
fig = plt.gcf()
fig.set_size_inches(10, 5)
plt.show()

# %%
df[[ 'Abortus','Systolic', 'Diastolic', 'Proteinuria', 'Delivery Method', 'Creatinine']].boxplot()
fig = plt.gcf()
fig.set_size_inches(10, 5)
plt.show()

# %%
df[['Hemoglobin','Leukocytes','Hematocrit', 'Platelets', 'Erythrocytes', 'Prognosis']].boxplot()
fig = plt.gcf()
fig.set_size_inches(10, 5)
plt.show()

# %%
# Replace Outliers for Height with Mean
df['Height'] = np.where(df['Height'] < 120, df['Height'].mean(), df['Height'])

df[['Height']].boxplot()
fig = plt.gcf()
fig.set_size_inches(6, 5)
plt.show()

# %%
# Replace Outliers for Diastolic with Mean
df['Diastolic'] = np.where(df['Diastolic'] < 60, df['Diastolic'].mean(), df['Diastolic'])

df[['Diastolic']].boxplot()
fig = plt.gcf()
fig.set_size_inches(6, 5)
plt.show()

# %%
# Concat Height & Weight as Body Mass Index
df['BMI'] =  df['Weight'] / ((df['Height'] / 100)**2)
# Drop Height & Body Weight
df.drop(['Height', 'Weight'], axis=1, inplace=True)
# Move BMI to Index number 1
df.insert(1, 'BMI', df.pop('BMI'))
df.head()

# %%
# Dataset correlation with sns heatmap
corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
plt.title('Correlation Matrix')
fig = plt.gcf()
fig.set_size_inches(12, 12)
plt.show()

# %% [markdown]
# ## Data Modeling

# %%
y = df['Prognosis']
X = df.drop('Prognosis', axis=1)

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# ### Decision Tree

# %%
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# %%
# Evaluate with classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# %%
# Plot Confusion Matrix with Heatmap
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
plt.title('Confusion Matrix Descision Tree')
fig = plt.gcf()
fig.set_size_inches(6, 6)
plt.show()

# %%
# Find the best Decision Tree model accuracy with KFold Cross Validation
from sklearn.model_selection import cross_val_score

k_values = [k for k in range(2, 11)]

for k in k_values:
    model = DecisionTreeClassifier(random_state=42)
    scores = cross_val_score(model, X, y, cv=k)
    # Evaluate Accuracy
    avg_accuracy = scores.mean()
    print(f"K = {k} | Average Scores = %.2f" % avg_accuracy)

# %% [markdown]
# ### Naive Bayes

# %%
from sklearn.naive_bayes import GaussianNB
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# %%
# Evaluate with classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# %%
# Plot Confusion Matrix with Heatmap
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
plt.title('Confusion Matrix Naive Bayes')
fig = plt.gcf()
fig.set_size_inches(6, 6)
plt.show()

# %%
# Find the best Naive Bayes model accuracy with KFold Cross Validation
k_values = [k for k in range(2, 11)]

for k in k_values:
    model = GaussianNB()
    scores = cross_val_score(model, X, y, cv=k)
    # Evaluate Accuracy
    avg_accuracy = scores.mean()
    print(f"K = {k} | Average Scores = %.2f" % avg_accuracy)


