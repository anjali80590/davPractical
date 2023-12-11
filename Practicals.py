# PRACTICAL-1

def list_of_dict(heights):
    keys=heights.keys()
    values = zip(*[heights[k] for k in keys]) 
    result = [dict(zip(keys,v )) for v in values]
    return result

heights = {'Boys':[72,68,70,69,74], 'Girls':[63,65,69,62,61]}
print("\n ORIGINAL DICTIONARY OF LISTS :" , heights)
print("\n NOW LIST OF DICTIONARIES : \n",list_of_dict(heights))


# PRACTICAL-2


import numpy as np

#part a
arr = np.random.randint(1,50,(4,6))
arr

#along the second axis
#Mean
print('Mean of the array: ',arr.mean(axis=1))
#standard deviation
print('Standard Deviation of the array: ',arr.std(axis=1))
#variance
print('Variance of the array: ',arr.var(axis=1))

#part b
B = [56, 48, 22, 41, 78, 91, 24, 46, 8, 33]
arr1 = np.array(B)
arr1

print("Sorted array: ",np.sort(arr1))
print("Indices of the sorted elements of a given array: ",np.argsort(arr1))

#part c
m = int(input('Enter the number of rows(m): '))
n = int(input('Enter the number of columns(n): '))
arr2 = np.random.randint(1,100,(m,n))
print(arr2)
print('Shape: ',arr2.shape)
print('Type: ',type(arr2))
print('Data Type: ',arr2.dtype)
arr2 = arr2.reshape(n,m)
print('After reshaping: \n',arr2)
print('New Shape: ',arr2.shape)

#part D
x = np.array([1, 0, 3, 4])
print("ORIGINAL ARRAY ::--> ",x)
print("\nTest if none of the elements of the said array is zero ::--> ", np.all(x))

res = np.where(x == 0)[0]
print("The index of the  zero elements is :: ",res)


x = np.array([1, 0, 0, 3, 2, 0])
print("\n-------------------------------------------------------")
print("\nORIGINAL ARRAY ::--> ",x)
print("\nTest whether any of the elements of a given array is non-zero ::",np.any(x))
res = np.where(x != 0)[0]
print("The index of the non- zero elements is :: ",res)
x = np.array([0, 0, 0, 0])


a = np.array([1, 0, np.nan, 3, np.nan])
print("\n-------------------------------------------------------")
print("\nORIGINAL ARRAY ::--> ",a)
print("\nTest element-wise for NaN :: ",np.isnan(a))

res = np.where(np.isnan(a) == True)[0]
print("The index of the  zero elements is :: ",res)


# PRACTIAL-3

import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randint(0,100,size=(50,3)), columns=list('123'))
df.head()

for c in  df.sample(int(df.shape[0]*df.shape[1]*0.10)).index:
    df.loc[c,str(np.random.randint(1,4))]=np.nan
df

#part A
print(df.isnull().sum().sum())

for col in df.columns:
    print(col,df[col].isnull().sum())
df.dropna(axis = 1,thresh=(df.shape[0]-5)).head()

sum=df.sum(axis=1)
print("SUM IS :\n",sum)
print("\nMAXIMUM SUM IS :",sum.max())
max_sum_row = df.sum(axis=1).idxmax()
print("\nRow index having maximum sum is :" ,max_sum_row)

df = df.drop(max_sum_row ,axis =0)
print("\nDATA Frame AFTER REMOVING THE ROW HAVING MAXIMUM SUM VALUE")
df

#part D
sortdf=df.sort_values('1')
sortdf.head()

# PART E
df =df.drop_duplicates(subset='1',keep = "first")
print(df)

#part F
correlation = df['1'].corr(df['2'])
print("CORRELATION between column 1 and 2 : ", correlation)
covariance = df['2'].cov(df['3'])
print("COVARIANCE between column 2 and 3 :",covariance)

#part G
df.plot.box()

#part H
df1 =  pd.cut(df['2'],bins=5).head()
df1


# PRACTICAL-4


import numpy as np
import pandas as pd

attdDay1 = pd.read_excel('Practical-4.xlsx',"Sheet1")
attdDay2 = pd.read_excel('Practical-4.xlsx',"Sheet2")

print(attdDay1.head(),"\n")
print(attdDay2.head())

#part A
pd.merge(attdDay1,attdDay2,how='inner',on='Name')

#part B
either_day = pd.merge(attdDay1,attdDay2,how='outer',on='Name')
either_day

#part C
either_day['Name'].count()

#part D
both_days = pd.merge(attdDay1,attdDay2,how='outer',on=['Name','Duration']).copy()
both_days.fillna(value='-',inplace=True)
both_days.set_index(['Name','Duration'])



# PRACTICAL-5


import pandas as pd
# 2d comprehsive plotting library creating static aniated interaive vis in py
import matplotlib.pyplot as plt
# data vistualisation library simplies creation informative atrib statical graphics
import seaborn as sns

# sepan leaf petal upper part 
# loads famous iris dataset form seaborn  into pandas dataframe named iris 
# containspeal legnth width petal width 150 iris flower 50 form 3 diff species 
iris = sns.load_dataset('iris')

# first five rows
iris.head()

#part A
sns.countplot(x='species',data=iris,palette='Set2')
plt.xlabel('Species')
plt.ylabel('Frequency')
plt.title('Frequency of Each class label')

#part B
# create scatter plot using matplb 
plt.scatter(x='petal_width',y='sepal_width',data=iris)
plt.xlabel('Petal Width')
plt.ylabel('Sepal Width')
plt.title("Scatter plot Petel width vs Sepal Width")

#part C
# creates a historam it plots distribution of petal length values from iris dataframe with 30 bins
sns.distplot(iris['petal_length'],kde=False,bins=30)

# create pair plot using seaboran r/b varible in irirs dataframe with diff color of each speicies
sns.pairplot(iris,hue='species',palette='coolwarm')


# PRACTICAL-6


import numpy as np
import pandas as pd

df = pd.read_csv('weather_by_cities.csv')
df.info()

df.groupby('city')['temperature'].mean()

df["day"].fillna(method = "ffill")

import datetime as dt
df['day'] = pd.to_datetime(df['day']).dt.strftime('%d-%m-%y')
df.head()

df_agg = df.groupby(['event']).agg({'temperature' : sum})
result = df_agg['temperature'].groupby(level = 0,group_keys= False)
print(result.nlargest())

weather=df.groupby(['event' , pd.cut(df.windspeed,10)])
result= weather.size().unstack()
print(result)


# PRACTICAL-7


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = [["Mudit Chauhan","December","M","III"], ["Seema Chopra","January","F","II"],
        ["Rani Gupta","March","F","I"], ["Aditya Narayan","October","M","I"], 
        ["Sanjeev Sahni","February","M","II"], ["Prakash Kumar","December","M","III"],
        ["Ritu Agarwal", "September","F","I"], ["Akshay Goel", "August", "M","I"], 
        ["Meeta Kulkarni" ,"July ","F", "II"],  ["Preeti Ahuja ","November", "F", "II"], 
        ["Sunil Das Gupta ","April", "M", "III"], ["Sonali Sapre ","January","F","I"],
        ["Rashmi Talwar", "June ","F" ,"III"],  ["Ashish Dubey" ,"May" ,"M", "II"], 
        ["Kiran Sharma", "February","F", "II"], ["Sameer Bansal", "October","M", "I" ]]

stu = pd.DataFrame(data=data, columns=["Name","Birth_Month","Gender","Pass_Division"])

stu.head()

gender = pd.get_dummies(stu['Gender'])
gender.head()

#Multi-collinearity
gender = pd.get_dummies(stu['Gender'],drop_first=True)
gender.head()

#Multi-collinearity
div = pd.get_dummies(stu['Pass_Division'])
div.head()

# part-2
stu.info()

months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
stu['Birth_Month'] = pd.Categorical(stu['Birth_Month'], categories=months, ordered=True)
stu.sort_values(by='Birth_Month',inplace=True)
stu.head()



# PRACTICAL-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = [["Shah","Male",114000.00],["Vats"," Male",65000.00],["Vats","Female",43150.00],["Kumar", "Female ",69500.00],
        ["Vats", "Female", 155000.00],["Kumar","Male", 103000.00],["Shah", "Male" ,55000.00],
        ["Shah", "Female", 112400.00],["Kumar","Female",81030.00],["Vats", "Male", 71900.00]]

family = pd.DataFrame(data=data, columns=["Name" ,"Gender", "MonthlyIncome"])
family.head()

family.info()

# PART-A
# group family dataframe by name coumn
familywise = family.groupby('Name')
# statiscs for each group and trnaspose result
familywise.describe().transpose()

familywise.sum()

# PART-B
familywise.max()

# part-c
family[family['MonthlyIncome']>60000]

# part-d
# group by name 
family[family['Gender']=='Female'].groupby('Name').mean().loc['Shah']