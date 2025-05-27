import pandas as pd
df = pd.read_csv("Titanic-Dataset.csv")
print(df.describe())
print(df.info())
print(df.isnull().sum())
import matplotlib.pyplot as plt
df.hist(bins=20, figsize=(10,8))
plt.tight_layout()
plt.show()
import seaborn as sns
sns.boxplot(x=df['Age'])
plt.title('Age Boxplot')
plt.show()
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()
sns.pairplot(df[['Survived', 'Age', 'Fare', 'Pclass']])
plt.show()
