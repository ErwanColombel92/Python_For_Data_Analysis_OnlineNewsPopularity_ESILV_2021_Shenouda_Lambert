# Python_For_Data_Analysis_OnlineNewsPopularity_ESILV_2021_Shenouda_Lambert

This work has been done by Gabriel Shenouda and Benjamin Lambert from ESILV - DIA1 - Paris in January 2021.

The notebook containing the analysis is called : "OnlineNewsPopularity_Analysis_Shenouda_Lambert"

# Dataset
Link : https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity

This dataset summarizes a heterogeneous set of features about articles published by Mashable in a period of two years.
The goal is to predict the number of shares in social networks (popularity).
— The dataset is composed of 61 attributes
— It has 39797 instances.

Attribute Information: 

     0. url:                           URL of the article 
     1. timedelta:                     Days between the article publication and 
                                       the dataset acquisition 
     2. n_tokens_title:                Number of words in the title 
     3. n_tokens_content:              Number of words in the content 
     4. n_unique_tokens:               Rate of unique words in the content 
     5. n_non_stop_words:              Rate of non-stop words in the content 
     6. n_non_stop_unique_tokens:      Rate of unique non-stop words in the 
                                       content 
     7. num_hrefs:                     Number of links 
     8. num_self_hrefs:                Number of links to other articles 
                                       published by Mashable 
     9. num_imgs:                      Number of images 
    10. num_videos:                    Number of videos 
    11. average_token_length:          Average length of the words in the 
                                       content 
    12. num_keywords:                  Number of keywords in the metadata 
    13. data_channel_is_lifestyle:     Is data channel 'Lifestyle'? 
    14. data_channel_is_entertainment: Is data channel 'Entertainment'? 
    15. data_channel_is_bus:           Is data channel 'Business'? 
    16. data_channel_is_socmed:        Is data channel 'Social Media'? 
    17. data_channel_is_tech:          Is data channel 'Tech'? 
    18. data_channel_is_world:         Is data channel 'World'? 
    19. kw_min_min:                    Worst keyword (min. shares) 
    20. kw_max_min:                    Worst keyword (max. shares) 
    21. kw_avg_min:                    Worst keyword (avg. shares) 
    22. kw_min_max:                    Best keyword (min. shares) 
    23. kw_max_max:                    Best keyword (max. shares) 
    24. kw_avg_max:                    Best keyword (avg. shares) 
    25. kw_min_avg:                    Avg. keyword (min. shares) 
    26. kw_max_avg:                    Avg. keyword (max. shares) 
    27. kw_avg_avg:                    Avg. keyword (avg. shares) 
    28. self_reference_min_shares:     Min. shares of referenced articles in 
                                       Mashable 
    29. self_reference_max_shares:     Max. shares of referenced articles in 
                                       Mashable 
    30. self_reference_avg_sharess:    Avg. shares of referenced articles in 
                                       Mashable 
    31. weekday_is_monday:             Was the article published on a Monday? 
    32. weekday_is_tuesday:            Was the article published on a Tuesday? 
    33. weekday_is_wednesday:          Was the article published on a Wednesday? 
    34. weekday_is_thursday:           Was the article published on a Thursday? 
    35. weekday_is_friday:             Was the article published on a Friday? 
    36. weekday_is_saturday:           Was the article published on a Saturday? 
    37. weekday_is_sunday:             Was the article published on a Sunday? 
    38. is_weekend:                    Was the article published on the weekend? 
    39. LDA_00:                        Closeness to LDA topic 0 
    40. LDA_01:                        Closeness to LDA topic 1 
    41. LDA_02:                        Closeness to LDA topic 2 
    42. LDA_03:                        Closeness to LDA topic 3 
    43. LDA_04:                        Closeness to LDA topic 4 
    44. global_subjectivity:           Text subjectivity
    45. global_sentiment_polarity:     Text sentiment polarity 
    46. global_rate_positive_words:    Rate of positive words in the content 
    47. global_rate_negative_words:    Rate of negative words in the content 
    48. rate_positive_words:           Rate of positive words among non-neutral 
                                       tokens 
    49. rate_negative_words:           Rate of negative words among non-neutral 
                                       tokens 
    50. avg_positive_polarity:         Avg. polarity of positive words 
    51. min_positive_polarity:         Min. polarity of positive words 
    52. max_positive_polarity:         Max. polarity of positive words 
    53. avg_negative_polarity:         Avg. polarity of negative  words 
    54. min_negative_polarity:         Min. polarity of negative  words 
    55. max_negative_polarity:         Max. polarity of negative  words
    56. title_subjectivity:            Title subjectivity 
    57. title_sentiment_polarity:      Title polarity 
    58. abs_title_subjectivity:        Absolute subjectivity level 
    59. abs_title_sentiment_polarity:  Absolute polarity level 
    60. shares:                        Number of shares (target) 

Sources :
K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision 
Support System for Predicting the Popularity of Online News. Proceedings 
of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence, 
September, Coimbra, Portugal. 

-- The articles were published by Mashable (www.mashable.com) and their content as the rights to reproduce it belongs to them. Hence, this dataset does not share the original content but some statistics associated with it. The original content be publicly accessed and retrieved using the provided urls. -- Acquisition date: January 8, 2015 --


# Experiments and results
We provide a set of visuals between the variables and the target 'shares'.

We tried to do regression on this dataset but the results were bad (MSE too high). 
We also stated that a multi class classification wasn't a good idea since the accuracy was also mediocre.
The reason for those results might be the lack of more data, models not performant enough, data pre-processing inneficient or other reasons.

We finally decided to make a simple classification where the target is the popularity, whether is is POPULAR or UNPOPULAR.

We tried to use a 3355 threshold (to decide whether the article is popular or not) which represents the mean of shares. 
We managed to get 80.3% of accuracy but the recall and F1-score were around 0, so we didn't consider it as a great model.

The threshold used is the median which is 1400.

We used and tunes differents models and scaled the data for some of them.

The models and the best accuracy we managed to get for each one of them :

Gradient Boosting : 66.97%

Random Forest : 66.67%

Ada Boost : 66.38%

Bagging : 66.20%

Logistic Regression : 65.05% with standard scaling

BernoulliNB : 63.79% with robust scaling

KNN : 63.73% with robust scaling

GaussianNB : 55.67% with standard scaling

You will find in the .ipynb the recall, precision, AUC, F1-score for each one of them.

# API

We alo provide an API using Flask. It works with an XGB model trained on the same dataset on the following parameters :

n_tokens_title',
'n_tokens_content',
'n_unique_tokens',
'average_token_length',
'n_non_stop_unique_tokens',
'num_hrefs',
'global_subjectivity', 
'avg_positive_polarity',
'global_sentiment_polarity', 'data_channel_is_world',
"data_channel_is_tech", "data_channel_is_socmed",
"data_channel_is_bus", "data_channel_is_entertainment",
"data_channel_is_lifestyle

"Model creation for Flask API.ipynb" is the file used to create the xgb model used it the API.

"Functions for Flask API.ipynb" contains the functions we used in order to make the model work in the API.

The accuracy of the model used in the API is 53.3%

# Commands to create and activate the virtual environment : 

In anaconda prompt we do :

mkdir flask_application

cd flask_application 

python -m venv .\monenv

from there the virtual environment is created

When we want to activate it (to test the code), we do on anaconda pronpt : 

cd flask_application 

monenv/Scripts/activate

python app2.py 

then we open a web page and put in the url "http://localhost:5000/"
