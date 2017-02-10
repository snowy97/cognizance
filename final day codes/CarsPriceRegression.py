from dateutil import parser
from sklearn import preprocessing, cross_validation, neighbors, svm, linear_model
from sklearn.ensemble import RandomForestRegressor
import time
import pandas as pd

df = pd.read_csv('autos.csv', encoding='cp1252')

print(df.head())
print(df.describe())
print(df.isnull().sum())


df['notRepairedDamage'].fillna(value='not-declared', inplace=True)
#df['gearbox'].fillna(value='not-declared', inplace=True)


df.dropna(inplace = True)
print(df.seller.unique())
print(df.offerType.unique())
print(df.abtest.unique())
print(df.nrOfPictures.unique())

print(df.groupby('seller').size())
df = df[(df['seller']!='gewerblich')]
df = df.drop(['nrOfPictures', 'name', 'offerType', 'seller'], 1)


df = df[(df.yearOfRegistration >= 1863) & (df.yearOfRegistration < 2017)]



ageCol = []
for date in df['dateCreated']:
    temp = parser.parse(date)
    ageCol.append(temp.year)


df['ageCol'] = ageCol - df['yearOfRegistration']

## Analysing, the age column, we find some random data so we can clean that

print(df['ageCol'].describe())

df = df[(df['ageCol'] >1) & (df['ageCol'] < 50)]

#Different ways of dropping columns
df.drop(['dateCreated'], 1, inplace = True)
df = df.drop(['monthOfRegistration', 'postalCode'], 1)


#Separating price before dropping
Z = df['price']
for cols in df.columns:
    df[cols] = preprocessing.LabelEncoder().fit_transform(df[cols])

## Can also try by using only these columns
#X = df.loc[:,["vehicleType","gearbox","powerPS","brand","model","fuelType"]]
X = df.drop(['price'], 1)

# Should see results with and without scaling
X = preprocessing.scale(X)


X_train, X_test, Z_train, Z_test = cross_validation.train_test_split(X,Z,test_size = 0.02)


model_sklearn = linear_model.LinearRegression()

start = time.time()
model_sklearn.fit(X_train, Z_train)
print(time.time() - start)
print(model_sklearn.score(X_test,Z_test))


print(df.columns)


##Using random forest
rf = RandomForestRegressor()

##recording time
start = time.time()
rf.fit(X_train, Z_train)
##Print time
print(time.time() - start)

## Random Forest gives importances to features
imortances = rf.feature_importances_
print(imortances)
print(rf.score(X_test,Z_test))



## Try and reduce the dataset enough so the svm classifier can be run
#
# sv = svm.SVR(kernel = 'linear', C = 10)
# sv.fit(X_train, Z_train)
# print(sv.score(X_test, Z_test))