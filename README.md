<H3>ENTER YOUR NAME : RAJESH A</H3>
<H3>ENTER YOUR REGISTER NO : 212222100042</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 28/02/2024</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```py
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
```
```py
df=pd.read_csv("/content/Churn_Modelling.csv", index_col="RowNumber")
df
```
```py
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df
```
```py
df.isnull().sum()
```
```py
df.duplicated()
```
```py
df.describe()
```
```py
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1
```
```py
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y
```
```py
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```

## OUTPUT:
### DATASET:
![nn1](https://github.com/Rajeshanbu/Ex-1-NN/assets/118924713/02a6ee63-035a-4f1b-bdb2-c34435c3a12a)

### DROPPING THE UNWANTED DATASET:
![nn2](https://github.com/Rajeshanbu/Ex-1-NN/assets/118924713/a5b0682d-5603-492b-a066-57cc38e631f9)

### CHECKING NULL VALUES:
![nn3](https://github.com/Rajeshanbu/Ex-1-NN/assets/118924713/0e93eebc-4282-400d-be9a-0fb754e769ae)


### CHECKING FOR DUPLICATION:
![nn4](https://github.com/Rajeshanbu/Ex-1-NN/assets/118924713/906b75bc-e03d-4634-aeea-ad6061830f66)


### DESCRIBING THE DATASET:
![nn5](https://github.com/Rajeshanbu/Ex-1-NN/assets/118924713/c4ad4b90-cf84-4a4a-9ce3-3798b675816a)

### SCALING THE DATASET:
![nn6](https://github.com/Rajeshanbu/Ex-1-NN/assets/118924713/731c3551-32dc-48e0-8ee1-0776196a816b)

### X FEATURES:
![nn7](https://github.com/Rajeshanbu/Ex-1-NN/assets/118924713/5f351a71-41dd-47ce-ae1d-948a917aa79d)


### Y FEATURES:
![nn8](https://github.com/Rajeshanbu/Ex-1-NN/assets/118924713/4f1b8df4-ba2a-45ec-b188-7e8d628c4d73)

### SPLITTING THE TRAINING AND TESTING DATASET:
![nn9](https://github.com/Rajeshanbu/Ex-1-NN/assets/118924713/1e635229-4ccc-4aeb-a97e-b3ade9fd2a65)


### RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


