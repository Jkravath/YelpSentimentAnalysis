# Yelp Bites: Predicting Restaurant Success Through Review Sentiment Analysis

## Abstract


This project aims to develop a machine learning model to classify Yelp reviews as positive, neutral, or negative, providing businesses with actionable insights. Using a subset of the Yelp Open Dataset comprising 40,000 reviews from Tampa restaurants, our team preprocessed the data through cleaning and lemmatization before applying models like Naive Bayes, Logistic Regression, and advanced techniques such as LDA, QDA, and PCA. Evaluation was performed using metrics like accuracy, precision, and recall, revealing that Logistic Regression was the most predictive model. This result underscores its effectiveness in sentiment analysis tasks, offering a practical tool for businesses to better understand customer feedback.


## Introduction 

Our project addresses the challenge of understanding customer feedback beyond simplistic star ratings. Businesses often rely on star ratings to gauge success, but these ratings can mask the true sentiment expressed in reviews. For instance, a customer might leave a five-star review but include textual feedback that reflects dissatisfaction. This gap in interpreting feedback highlights the need for a sentiment analysis model that can classify reviews as positive, neutral, or negative to provide a clearer picture of customer experiences.

This problem is interesting because it bridges natural language processing (NLP) with practical business applications. By analyzing sentiments in textual reviews, businesses can gain nuanced insights to improve their services, resolve customer issues, and maintain competitive edges. The broader use cases extend beyond restaurants to other service-oriented industries, enabling companies to make data-driven decisions based on customer sentiment.

Our approach leverages machine learning techniques to analyze a subset of the Yelp Open Dataset, focusing on approximately 40,000 reviews from Tampa-based restaurants. We explored models ranging from simple Naive Bayes and Logistic Regression to advanced techniques like Linear Discriminant Analysis, Quadratic Discriminant Analysis, and Principal Components Analysis. These methods convert textual data into numerical vectors, enabling fast and accurate classification. We analyzed these models based on the metrics of accuracy, precision, and recall. 

Logistic Regression emerged as the most predictive model in our evaluation, offering an optimal balance of simplicity, computational efficiency, and accuracy. Compared to more complex models, it effectively handled the text data's high dimensionality without overfitting. Previous studies have explored sentiment analysis using similar approaches, but our work emphasizes the practical application of Logistic Regression in a real-world dataset, with robust preprocessing and evaluation ensuring reliable results.

The key components of our approach include thorough data cleaning and lemmatization to prepare text data, applying diverse machine learning models, and evaluating them against a consistent set of criteria. While our method achieved strong results, it is not without limitations. For instance, the model may struggle with nuanced sentiments or sarcasm, which are difficult to detect in text data. Despite these challenges, our approach demonstrates the value of sentiment analysis in business intelligence and sets the stage for future work to refine and expand its application.

## Setup 

Our project analyzes a subset of the Yelp Open Dataset to predict the sentiment of restaurant reviews. The dataset consists of approximately 40,000 reviews from restaurants in Tampa, with key columns including the text of the reviews and their associated star ratings. Basic statistics of the dataset reveal an average word count of 97.7, an average character count of 528, an average word length of 4.46 and an average star rating of 3.804. We also analyzed stopwords in each review, which in Natural Language Processing, represents words that don't carry any meaning and will not be useful for the model. We found for the reviews analyzed, there was an average stopword count of 45.75 and an average stopword rate of 0.448. Data preprocessing included converting all text to lowercase, removing punctuation and stopwords, and applying lemmatization to standardize word forms. Lemmatization identifies each word's intended meaning and reduces it down to its base form, this step is necessary for our models to best interpret the sentiment of each word in a review. Our preprocessing ensured that the text data was cleaned and ready for use in machine learning models.

The experimental setup includes a range of machine learning models for text classification. We started with simpler models like Naive Bayes and Logistic Regression and then tested advanced techniques such as Linear Discriminant Analysis, Quadratic Discriminant Analysis , and Principal Components Analysis. These models were evaluated based on performance metrics such as accuracy, precision, and recall. To ensure robust evaluations, we used cross-validation to assess model performance and estimate variability in the results.

Our implementation transformed text into numerical representations using techniques such as word vectorization, enabling models to process the data effectively. For computational efficiency, we employed feature reduction techniques, such as Principal Component Analysis (PCA), to minimize the dimensionality of the dataset. We found that the optimal number of principle components is 19 as that yielded the lowest cross-validated MSE.

The computing environment for our experiments included Python libraries such as scikit-learn for machine learning implementation and evaluation, Pandas and Numpy for data manipulation, and Matplotlib for visualizing results. These experiments were conducted on a standard personal computer with sufficient memory and processing power for the dataset size. This setup provided a scalable and accessible framework for testing various machine learning models while addressing the challenges posed by high-dimensional text data.

## Results

As discussed before, each model was evaluated along the metrics of precision, recall, F1-score, and accuracy. 

_Logistic Regression_: Performed well across the board with a weighted average of 0.94 for precision, recall, F1-Score and accuracy. The model did extremely well identifying neutral or positive reviews. The model did less well analyzing negative reviews, having a significantly lower F1-Score and recall. However, as discussed later, Logistic Regression displays the best performance across all of the metrics. 

<img width="518" alt="Screenshot 2024-12-08 at 6 09 14 PM" src="https://github.com/user-attachments/assets/f331d1ec-e71e-4550-a57b-a1cfdeaf0529">

_Naive Bayes_: Naive Bayes had mixed results. While accuracy and precision of naive bayes stayed relatively consistent along the categories of negative, neutral and positive, recall and F1-score varied dramatically. For negative reviews, the model has a very low recall and F1-score, hovering around 0.05-0.1.

![image](https://github.com/user-attachments/assets/85840a9e-7950-44c3-8ca4-da00cd80b86a)

_PCA_: We perfomed PCA before running more complex models in order to reduce the dimensionality of the data. Initial PCA testing suggested 20 as the number of principal components for the model. However, after using cross-validation to improve the model, we found the optimal number of components between the range of 1-21 was 19. Therefore, we used this PCA model with 19 components to perform the more complex models of LDA and QDA. 

![image](https://github.com/user-attachments/assets/50c7e53d-6c8f-4512-9c39-e4e1a8f93d44)

_LDA_: LDA performed badly predicting negative reviews, with low scores. It performed decently well with neutral reviews, and well with positive reviews. '

![image](https://github.com/user-attachments/assets/45e6eb7a-a686-4c5a-bc03-1df577f09908)

_QDA_: QDA performed similarly to LDA with bad performance across metrics, except for accuracy, on the negative reviews. It performs better on neutral and positive reviews.

![image](https://github.com/user-attachments/assets/930145d7-b96d-4dc9-8c3e-d2498860e31f)


## Discussion

After examining all 4 models, we find the weighted average of the important metrics on the table below. 

<img width="460" alt="Screenshot 2024-12-07 at 6 33 54 PM" src="https://github.com/user-attachments/assets/148408c3-f4f0-4e8e-b9d6-47b408736310">

Observe how Logistic Regression performs the best across every metric observed. Additionally, Logistic Regression performed the best among each model _in each category_ to predict negative, neutral, and positive reviews. 

Another note, is that each model performed the worst on negative reviews. This is likely due to the fact that there were far less negative reviews for the models to analyze than neutral and positive reviews. Yelp reviews tend to lean toward the positive, meaning there are much more positive reviews than negative ones. This also explains why each model performed the best predicting positive reviews as well. 

In the future, these models should be trained with more negative reviews so that they are more able to accurately predict which words predict a negative sentiment. Furthermore, reviews that are sarcastic or funny can be hard for these models to interpret, future models should integrate understandings of this. 



## Conclusion

The goal of this project was to find the best model to predict customer sentiment from a text review. The Logistic Regression model far and away beats the other models, even those that are more complex. We believe Logistic Regression performed the best because it is a simple, interpretable model that can efficiently handle high-dimensional text data through techniques like TF-IDF and regularization. Its ability to make linear predictions and manage imbalanced data with class weighting makes it effective for distinguishing between positive and negative reviews. 

## References
