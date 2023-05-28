#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data_train = pd.read_csv(r'D:\Semester 5\Secure Software Systems\Assignment\new_bard\Network security data set\KDDTrain+.txt', sep = ',', encoding = 'utf-8', header = None)


# In[3]:


data_test = pd.read_csv(r'D:\Semester 5\Secure Software Systems\Assignment\new_bard\Network security data set\KDDTest+.txt', sep = ',', encoding = 'utf-8', header = None)


# In[4]:


import matplotlib.pyplot as plt


# ## Identifying and Handling Imputations

# In[5]:


data_train.info()


# In[6]:


data_train.isna().sum()


# In[7]:


data_test.info()


# In[8]:


data_test.isna().sum()


# ## Identifying and handling outliers

# In[9]:


numerical_train = data_train.select_dtypes(['int64', 'float64'])


# In[10]:


for col in numerical_train.columns:
    dataset = data_train.copy()
    dataset.boxplot(col)
    plt.ylabel('Count')
    plt.title(col)
    plt.show()


# In[11]:


for col in numerical_train.columns:
    dataset = data_train.copy()
    plt.scatter(dataset[col], dataset[42])
    plt.xlabel(col)
    plt.ylabel('attack_state')
    plt.title(col)
    plt.show()


# ## Feature Engineering

# In[12]:


y_train = data_train[42]
y_test = data_test[42]


# In[13]:


categorical_train = data_train.select_dtypes('object')


# In[14]:


categorical_train


# In[15]:


for col in categorical_train.columns:
    print(categorical_train[col].unique())


# In[16]:


categorical_test = data_test.select_dtypes('object')


# In[17]:


for col in categorical_test.columns:
    print(categorical_test[col].unique())


# In[18]:


data_train = data_train.drop(labels = [42], axis = 1)


# In[19]:


data_test = data_test.drop(labels = [42], axis = 1)


# In[20]:


data_train.shape


# In[21]:


data_test.shape


# In[22]:


dataset = pd.concat([data_train, data_test], axis = 0)


# In[23]:


dataset.info()


# In[24]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# In[25]:


ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_train.columns)], remainder='passthrough')
dataset = ct.fit_transform(dataset)


# In[26]:


dataset


# In[27]:


from scipy.sparse import csr_matrix
dataset = pd.DataFrame.sparse.from_spmatrix(dataset)


# In[28]:


data_train = dataset.iloc[:125973, :]
data_test = dataset.iloc[125973:, :]


# In[29]:


data_train


# In[30]:


data_test


# In[31]:


dataset.info()


# In[32]:


dataset.select_dtypes(['object']).info()


# ## Scaling the data

# In[33]:


x_train = data_train
x_test = data_test


# In[34]:


x_train


# In[35]:


from sklearn.preprocessing import Normalizer


# In[36]:


scaler = Normalizer()


# In[37]:


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[38]:


x_train.shape


# In[39]:


x_train


# In[40]:


y_train[:10]


# In[41]:


x_train = x_train.toarray()


# In[42]:


x_test = x_test.toarray()


# In[43]:


x_test.shape


# In[44]:


x_test


# In[45]:


y_train = y_train.values
y_train = y_train.reshape(-1,1)


# ## ANN

# In[46]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()

classifier.add(keras.Input(shape = 162))

classifier.add(Dense(units = 80, kernel_initializer = "uniform", activation = "relu"))
classifier.add(Dropout(rate = 0.1))

classifier.add(Dense(units = 80, kernel_initializer = "uniform", activation = "relu"))

classifier.add(Dense(units = 80, kernel_initializer = "uniform", activation = "relu"))

classifier.add(Dense(units = 1, kernel_initializer = "uniform"))

classifier.compile(optimizer = 'Adamax', loss = keras.losses.MeanSquaredError())


# In[47]:


classifier.fit(x_train, y_train, batch_size = 50, epochs = 10)


# In[48]:


y_train


# In[49]:


y_pred = classifier.predict(x_test)


# In[50]:


y_pred


# In[51]:


y_test


# In[52]:


y_t = pd.DataFrame(y_test)
y_t = y_t.reset_index()
y_p = pd.DataFrame(y_pred)


# In[53]:


y = pd.concat([y_t, y_p], axis = 1)


# In[54]:


y = y.rename(columns={42:'y_test', 0:'y_pred'})


# In[55]:


from scipy import stats
from scipy.stats import norm
import seaborn as sns


# In[56]:


plt.figure(figsize=(10,10))
sns.distplot(y['y_test'] , fit=norm);

(mu, sigma) = norm.fit(y['y_test'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('y_test distribution')
plt.show()


# In[57]:


plt.figure(figsize=(10,10))
sns.distplot(y['y_pred'] , fit=norm);

(mu, sigma) = norm.fit(y['y_pred'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('y_pred distribution')
plt.show()


# In[58]:


y.to_csv('y.csv', index=False)


# In[59]:


columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot',
            'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations',
            'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count',
            'serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
            'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
            'dst_host_srv_rerror_rate','attack'])


# In[60]:


input_array = []

for i in range(len(columns)):
    temp = input('{} : '.format(columns[i]))
    input_array.append(temp)


# In[61]:


input_array = pd.DataFrame(input_array).T


# In[62]:


input_array


# In[63]:


input_array = ct.transform(input_array)


# In[64]:


input_array = scaler.transform(input_array)


# In[65]:


input_array


# In[66]:


input_array = input_array.toarray()


# In[67]:


prediction = classifier.predict(input_array)


# In[68]:


print('Attaxk_State : ', prediction)


# In[ ]:




