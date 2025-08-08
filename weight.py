import pandas as pd
import statsmodels.api as sm

data = pd.DataFrame({
    'Color': ['Red', 'Green', 'Blue', 'Red', 'Blue', 'Green'],
    'Price': [10, 12, 14, 11, 15, 13]
})
X = pd.get_dummies(data['Color'], drop_first=True)
X = X.astype(float)
X = sm.add_constant(X)
y = data['Price']
model = sm.OLS(y, X).fit()
print(model.summary())
