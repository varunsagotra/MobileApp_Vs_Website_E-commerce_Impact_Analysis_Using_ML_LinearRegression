#!/usr/bin/env python
# coding: utf-8

# # MobileApp_Vs_Website_EcommerceSaleAnalysis
# ### Using_ML_LinearRegression
# 
# `This is a Micro ML project`
# > Small projects demonstrate how we can use scikit-learn to create ML models in Python, dealing with a variety of datasets. 
# 
# - For this project, we have a (fake) dataset from a (fake) Ecommerce company that sells clothing online but also has in-store style and clothing.
# 
# `Use Case :: Company wants to decide whether to focus their efforts on their mobile app experience or their website, depending on which one of them has the greater impact`
# 
# Let's try to answer their question.

# In[1]:


# Import all required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# `# DataSet`
# 
# The dataset contains customer info. such as Email Address, and their Avatar. But as this is a regression project, we'll deal with the numerical features we have: 
# 
# **Avg. Session Length**: Average session of in-store style advice sessions
# **Time on App**: Average time spent on App in minutes
# **Time on Website**: Average time spent on Website in minutes
# **Length of Membership**: How many years the customer has been a member
# 

# In[2]:


customers = pd.read_csv('/Users/ceo/Desktop/Ecommerce Customers')


# In[3]:


customers.head()


# In[4]:


customers.describe()


# In[5]:


customers.info()


# ## Exploratory Analysis
# 
# Before we begin fitting a linear regression model on the data, let's try and visualize it first.
# 
# `Visualising the relationship between time spent on Website and yearly spend`

# In[25]:


sns.jointplot(customers['Time on Website'],customers['Yearly Amount Spent'])
plt.show()


# Visualising the relationship between time spent on app, and yearly spend.

# In[26]:


sns.jointplot(customers['Time on App'],customers['Yearly Amount Spent'])
plt.show()


# `Observation` :: Just from the above two visuals, we can conclude that there's a stronger correlation between time spent on app, and the yearly spend, than time spent on the website.
# 
# **Let's visualise the relationship between the different variables using a seaborn pairplot.

# In[8]:


sns.pairplot(customers)


# In[9]:


customers.corr()


# `Observation`: It looks like the length of membership is the feature that's the most (positively) correlated with yearly amount spent. This makes sense, as loyal customers are inclined to spend more.
# 
# We can use seaborn to fit this on a linear plot.

# In[27]:


sns.lmplot('Length of Membership','Yearly Amount Spent',data=customers)
plt.show()


# ## Splitting the Data
# 
# We're going to split the data between training and test sets, in a 70:30 ratio.

# In[11]:


customers.columns


# In[12]:


#Selecting only the numerical features for training the model.
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X_test,X_train,y_test,y_train = train_test_split(X,y,test_size=0.3,random_state=101)


# ## Training the Model

# In[15]:


from sklearn.linear_model import LinearRegression


# In[16]:


lm = LinearRegression()


# In[17]:


lm.fit(X_train,y_train)


# ## Making Predictions
# 

# In[18]:


predictions = lm.predict(X_test) 


# ** Create a scatterplot of the real test values versus the predicted values. **

# To visualise the predictions, let's create a scatterplot between real and predicted values.

# In[19]:


plt.scatter(y_test,predictions)


# `Observation` : Nice, it looks like our model performs fairly well, as our predictions and real values fit linearly without much variation.

# ## Evaluation and Understanding Results
# 
# But there's a standard way to evaluate linear regression models. Let's calculate the residual sum of squares.

# In[20]:


from sklearn import metrics


# In[21]:


print('MAE:',metrics.mean_absolute_error(y_test,predictions))
print('MSE:',metrics.mean_squared_error(y_test,predictions))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,predictions)))


# Let's try to interpret the coefficients for the variables.

# In[22]:


cust_coeff = pd.DataFrame(lm.coef_,X.columns)
cust_coeff.columns = ['Coefficient']
cust_coeff


# `What the coefficients mean`, is that, assuming all other features stay fixed,
# 
# - 1 unit increase in the Avg. Session Length leads to an approximate \$25 increase in yearly spend.
# - 1 unit increase in the Time on App leads to an approximate \$39 increase in yearly spend.
# - 1 unit increase in the Time on Website leads to an approximate \$0.77 increase in yearly spend.
# - 1 unit increase in the Length of Membership leads to an approximate \$62 increase in yearly spend.
# 

# ## App or Website? 
# 
# **So should the company focus more on their mobile app or on their website?**

# `**Conclusion**`
# > Between the two, the **mobile app seems to be doing better than the website**, as we see a greater increase in the Yearly amount spent with an increase in the time spent on the app (as opposed to the marginal increase on with time on website). 
# 
# `So there are two ways to approach the problem:`
# 
# - The company either focuses on the website to have it catch up in terms of the mobile app. Or,
# - They can focus on the mobile app, to maximise the benefits.
# 
# What we could also explore is the relationship between length of membership, and the time on app or website, as the length of membership seems to be more important for yearly spend.

# In[ ]:




