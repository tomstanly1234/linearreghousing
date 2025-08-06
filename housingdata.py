import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
housing=pd.read_csv('housing.csv')
housing=housing.dropna()
correlation_matrix=housing.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
oceanproximity=housing['ocean_proximity']
le=LabelEncoder()
speciesbynum=le.fit_transform(oceanproximity)
housing.drop('ocean_proximity',axis=1)
housing['ocean_proximity']=speciesbynum
features=['latitude','total_bedrooms','median_income','ocean_proximity']
scaler=StandardScaler()
housing[features]=scaler.fit_transform(housing[features])
X=housing[features].to_numpy()
m=X.shape[0]
X=np.c_[np.ones(m),X]
weights=np.random.randn(X.shape[1]).astype(np.float64)
var_ocean=speciesbynum.tolist()
learningrate=0.01
y=housing['median_house_value'].to_numpy()
for values in range(1000):
    ypred=X@weights
    error=(y-ypred).flatten()
    gradients=-(2/m)*X.T@error
    weights-=learningrate*gradients
print(weights)
