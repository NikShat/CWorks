#!/usr/bin/env python
# coding: utf-8

# In[61]:


import os
from ludwig.api import LudwigModel
import pandas as pd
import numpy as np


# ##### Сохранены некоторые переменные путей файлов для дальнейшего использования

# In[62]:


path_yes = r'.\brain_tumor_dataset\yes'
path_no = r'.\brain_tumor_dataset\no'
a = os.getcwd()


# ##### Собраны пути к файлам из искомых двух папок

# In[63]:


yes = []
for root, dirs, files in os.walk(path_yes):
    for file in files:
        part_path = r"/brain_tumor_dataset/yes/"
        sum = a + part_path + file
        yes.append(sum)
no = []
for root, dirs, files in os.walk(path_no):
    for file in files:
        part_path1 = r"/brain_tumor_dataset/no/"
        sum1 = a + part_path1 + file
        no.append(sum1)


# In[64]:


df = pd.DataFrame({'Brain_image': yes + no}) # brain_image -- сканы МРТ головного мозга 
df.index = np.arange(1, len(df)+1)
a = [1]*(len(yes))+[0]*(len(no)) # 0 -- отсутствие опухоли мозга 1 -- наличие опухоли мозга
df.insert(1, "tumor", a, True)
df.columns = ['Brain_image', 'Tumor']
df


# In[65]:


df.to_csv('result.csv')
train_data = pd.read_csv('result.csv')


# ##### Описание модели

# In[66]:


model_definition = { 
    "input_features": [  
        {"name": "Brain_image", "type": "image", "preprocessing":
         {"height": 128,
          "width": 128,
          "resize_method": "interpolate",
          "scaling": "pixel_normalization",
         "num_channels": 1}
        }
    ],  
    "output_features": [
        {"name": "Tumor", "type": "numerical"}
    ]
}


# In[67]:


model = LudwigModel(model_definition)


# In[68]:


train_stats = model.train(train_data) # тренировка модели


# In[69]:


model.save("braincheck") #модель сохранена

