## Authors
Tian Yi Xia<sup>1</sup>, Vlad Marinescu<sup>1</sup>, MinSeo Hur<sup>1</sup>, Ashwini Adhikari<sup>1</sup>, Theodore Philipe<sup>2</sup>, Jessica Liddell<sup>2</sup>   

<sup>1</sup>Youreka Montreal, <su

## Abstract

Antibiotic resistance (ABR) is a growing global health concern that threatens the future of human wellbeing. The leading causes of increased ABR—excessive and inappropriate use of antibiotics—can be linked to a diverse array of multidisciplinary environmental factors, largely those related to the healthcare, agricultural, and food industries. Here, we employ machine learning to understand antibiotic resistance trends, utilizing a dataset of 804 socioeconomic, environmental, and demographic indicators to identify the most effective predictors of and factors contributing to ABR, and predict antibiotic resistance for years with available data using the model. We predicted that machine learning would be able to accurately predict antibiotic resistance and that the significant predictors would be diverse, though primarily environmental-related. We found that global antibiotic use and resistance risk has predictables trends by means of machine learning, with our models’ accuracy being above 95%. We also found that the best predictors of ABR were most often environmental factors. Nonetheless, the significance of a diverse array of factors indicates that antibiotic use and resistance is an inherently multidisciplinary issue. Overall, our study can be used to inform policymakers across all disciplines to implement measures to mitigate the rise of antibiotic resistance.


## Results
### Confusion matrices in predicting total antibiotic usage
#### Predicted and actual total antibiotic usage category confusion matrix for training data
- 2003 to 2017

(Click image to enlarge)

[![Confusion matrix of testing prediction](https://raw.githubusercontent.com/ThatAquarel/health/total_antibiotic_usage/prediction/results/2003-2017_confusion_matrix_pred.png)](https://raw.githubusercontent.com/ThatAquarel/health/total_antibiotic_usage/prediction/results/2003-2017_confusion_matrix_pred.png)

#### Predicted and actual total antibiotic usage category confusion matrix for testing data
- 2018

(Click image to enlarge)

[![Confusion matrix of testing prediction](https://raw.githubusercontent.com/ThatAquarel/health/total_antibiotic_usage/prediction/results/2018_confusion_matrix_pred.png)](https://raw.githubusercontent.com/ThatAquarel/health/total_antibiotic_usage/prediction/results/2018_confusion_matrix_pred.png)

### Correspondence between top significant indicators and total antibiotic usage
Worldwide analysis of 2022

#### Top significant indicators, all categories
- n=100 indicators
- n=145 countries

(Click image to enlarge)

[![Clustermap of top 100 indicators](https://raw.githubusercontent.com/ThatAquarel/health/total_antibiotic_usage/visualizations/heatmap_top100.png)](https://raw.githubusercontent.com/ThatAquarel/health/total_antibiotic_usage/visualizations/heatmap_top100.png)

#### Top significant indicators, environmental
- n=87 environmental and climate change indicators
- n=145 countries

(Click image to enlarge)

[![Clustermap of top 100 environmental indicators](https://raw.githubusercontent.com/ThatAquarel/health/total_antibiotic_usage/visualizations/heatmap_env.png)](https://raw.githubusercontent.com/ThatAquarel/health/total_antibiotic_usage/visualizations/heatmap_env.png)

### Prediction of total antibiotic usage and risk of resistance, category by country
- 2022

(Click image to interact)

[![Predictions map](https://raw.githubusercontent.com/ThatAquarel/health/total_antibiotic_usage/docs/map.PNG)](https://thataquarel.github.io/health/predictions.html)

## Paper

[Machine Learning for Worldwide Antibiotic Usage and Resistance Prediction: A Longitudinal Study on Effective Environmental, Economical, and Social Predictors](https://github.com/ThatAquarel/health/blob/total_antibiotic_usage/docs/manuscript.pdf)
