#!/usr/bin/env python
# coding: utf-8

# In[88]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np


# In[89]:


df = pd.read_csv("africa.csv")
df


# In[90]:


# колонка с категориальными переменными
df["banking_crisis"].unique()


# ####### "crisis" = 1
# ####### "no_crisis" = 0

# In[32]:


# перевод в численные значения 0 и 1
df["banking_crisis"] = df["banking_crisis"].apply(lambda x: ["crisis", "no_crisis"].index(x))


# In[33]:


df.columns


# In[ ]:


# Для дальнейшего формирования данных для модели исключаются колонки: 
# case -- число, присвоенное стране, 
# cc3 -- трёхбуквенный код страны, 
# country -- название страны (так как первая модель строилась для одной страны, 
# year -- год исследования


# In[35]:


import re


# In[43]:


# выделение из всех данных страны из условия -- Ivory Coast
pat = r'Ivory Coast'
df2 = df[df['country'].str.contains(pat)]
df2


# In[59]:


enc = OneHotEncoder(categories='auto')
enc.fit(df2[df.columns[5:14]])


# In[91]:


# перевод данных в массив
df3 = enc.transform(df2[df.columns[5:14]]).toarray()


# In[64]:


# модель предсказывает наличие (1) или отсутствие (0) системного кризиса
Y = df2["systemic_crisis"].values


# In[65]:


# разделение данных на обучающие выборки
train_X, test_X, train_Y, test_Y = train_test_split(df3, Y)


# In[66]:


# в качестве модели выбрана логистическая регрессия
model = LogisticRegression()


# In[67]:


# обучение модели
model.fit(train_X, train_Y)


# In[68]:


# Предсказание системного кризиса в Ivory Coast
test_Yhat = model.predict(test_X)


# In[97]:


# Проверка точности
accuracy_score(test_Y, test_Yhat)


# In[98]:


# Проверка точности вторым способом
balanced_accuracy_score(test_Y, test_Yhat)


# ### Ответы на вопросы

# In[87]:


pd.crosstab(df["country"], df["systemic_crisis"])


# #### Больше всего системных кризисов произошло в Центральной Африканской республике

# In[83]:


pd.crosstab(df["country"], df["banking_crisis"])


# #### Больше всего кризисов банковской системы произошло в Египте

# In[84]:


pd.crosstab(df["country"], df["inflation_crises"])


# #### Больше всего инфляционных кризисов произошло в Анголе

# ##### Информации по ВВП в датасете нет, ответить на второй вопрос из условия затруднительно
