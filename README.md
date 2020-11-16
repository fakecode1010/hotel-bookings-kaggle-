# hotel-bookings-kaggle-
Classification models for High Dimensional Data
# Abstract
this project aims to create a model to predict the cancellation of hotel bookings accross 31 predictors ,such as arrival date, meal type, is_repeating_customer , and number_of_adults.
We noticed that the features contains multicolinearity and thus iterative SIS- Lasso (L1 penalty) method was used to screened out variables.
# Instructions
the code contains 4 section :
* iterative SIS method with d predictors as the output
* LOGIT model
* Decision Tree model
* KNN model
each models should run after iterative SIS lasso has been run. note that the dataset requires cleanup as there are 'NA' values. in additoin, dummy variables were used to categorical vairables with many class in particular , the variable country.

# Credits
* Chua Kang Wei
* Frederic Liew Yong Lun
* Lee yong jie , Richard
* Nicholas Alexander\
the dataset used was obtained from kaggle :https://www.kaggle.com/jessemostipak/hotel-booking-demand

# Discussion
the logit model used Youden criterion to mamixise both the sensitivty and specitivity by selecting the best threshold. 
The KNN model has the best accuracy since data behave similarly. Note that the number of features selected was 9 out of 31.The more features are included the more sparse the data is scatttered and lower KNN performance will be obtained.

