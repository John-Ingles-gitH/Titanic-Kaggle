import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from patsy import dmatrices
import statsmodels.api as sm
import seaborn as sns

#load dataset
titanic_data = pd.read_csv('C:/Users/John Ingles/Project2_John_Ingles/titanic-data.csv')

titanic_data.describe()

titanic_data_age_cleaned = titanic_data.dropna(subset=["Age"])

sns.set(style="ticks", color_codes=True)

sns.pairplot(titanic_data_age_cleaned, vars= ("Pclass","Age"), size=3, hue="Sex")
plt.suptitle("Pairplot of Age and Class grouped by Sex")

#plots a kernel density estimate of the passengers' age
titanic_data.Age.plot(kind='kde')
plt.title("Age Distribution Onboard")
plt.xlabel("Age")

sns.violinplot(x="Pclass",y="Age", hue="Sex", data=titanic_data, dropna=True)
plt.title("Age Distribution of each Class")

def means_grouped_by_x(data, x):
    return data.groupby(x).mean()
	
means_grouped_by_x(titanic_data,"Pclass")

means_grouped_by_x(titanic_data, "Sex")

means_grouped_by_x(titanic_data,"Survived")

class_total = pd.crosstab(titanic_data.Pclass, titanic_data.Survived)
class_surv = pd.crosstab(titanic_data.Pclass, titanic_data[titanic_data.Survived==1].Survived)
class_surv.div(class_total.sum(1).astype(float), axis=0).multiply(100)

class_surv.div(class_total.sum(1).astype(float), axis=0).multiply(100).plot(kind='bar');
plt.title("Percent Survived of each Class")
plt.ylabel("Percent Survived")

sex_total = pd.crosstab(titanic_data.Sex, titanic_data.Survived)
sex_surv = pd.crosstab(titanic_data.Sex, titanic_data[titanic_data.Survived == 1].Survived)
sex_surv.div(sex_total.sum(1).astype(float), axis=0).multiply(100)

sex_surv.div(sex_total.sum(1).astype(float), axis=0).multiply(100).plot(kind='bar')
plt.title("Percent Survived of each Sex")
plt.ylabel("Percent Survived")

survivors_group = titanic_data[titanic_data.Survived == 1]

class_sex_total = pd.crosstab(titanic_data.Pclass, titanic_data.Sex)
class_sex_survivors = pd.crosstab(survivors_group.Pclass, survivors_group.Sex)

class_sex_survivors.div(class_sex_total, axis=0).multiply(100).plot(kind='bar')
plt.title("Percent Survived by Class and Sex")
plt.ylabel("Percent Survived")

#define variables for Logit, C indicates categorical variables and variable to the left of ~ indicates the dependant variable
formula = 'Survived ~ C(Pclass) + C(Sex) + Age'

# create a regression friendly dataframe using patsy's dmatrices function
y,x = dmatrices(formula, data=titanic_data_age_cleaned, return_type='dataframe')

# instantiate our model
model = sm.Logit(y,x)

# fit our model to the training data
results = model.fit()

#print result summary
print (results.summary())
print ("\n")
print ('Change in odds of survival for a one unit change in parameter')
params = results.params
conf = results.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OR']
print (np.exp(conf))