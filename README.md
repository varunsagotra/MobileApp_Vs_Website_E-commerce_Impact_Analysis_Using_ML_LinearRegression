# MobileApp_Vs_Website_EcommerceSaleAnalysis
### Using_ML_LinearRegression

`This is a Micro ML project`
> Small projects demonstrate how we can use scikit-learn to create ML models in Python, dealing with a variety of datasets. 

- For this project, we have a (fake) dataset from a (fake) Ecommerce company that sells clothing online but also has in-store style and clothing.

`Use Case :: Company wants to decide whether to focus their efforts on their mobile app experience or their website, depending on which one of them has the greater impact`

> Let's try to answer their question

## Evaluation and Understanding Results
MAE: 8.277224105585296
MSE: 109.36337929836583
RMSE: 10.457694741116027

## Interpret the coefficients for the variables

Coefficient
Avg. Session Length	25.114639
Time on App	39.022188
Time on Website	0.767136
Length of Membership	62.247287

`What the coefficients mean`, is that, assuming all other features stay fixed,

- 1 unit increase in the Avg. Session Length leads to an approximate \$25 increase in yearly spend.
- 1 unit increase in the Time on App leads to an approximate \$39 increase in yearly spend.
- 1 unit increase in the Time on Website leads to an approximate \$0.77 increase in yearly spend.
- 1 unit increase in the Length of Membership leads to an approximate \$62 increase in yearly spend.

## App or Website? 

**So should the company focus more on their mobile app or on their website?**

`**Conclusion**`
> Between the two, the **mobile app seems to be doing better than the website**, as we see a greater increase in the Yearly amount spent with an increase in the time spent on the app (as opposed to the marginal increase on with time on website). 

`So there are two ways to approach the problem:`

- The company either focuses on the website to have it catch up in terms of the mobile app. Or,
- They can focus on the mobile app, to maximise the benefits.

## Next Analysis in bucket >>>
`What we could also explore is the relationship between length of membership, and the time on app or website, as the length of membership seems to be more important for yearly spend`
