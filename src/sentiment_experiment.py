from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

<<<<<<< HEAD:to_sort/src/initial_models/sentiment_experiment.py
from src.data_collection import get_article_by_name
=======
from to_sort.get_article_by_name import get_article_by_name
>>>>>>> a96341abddd6e09186592d5da09defe66b974bb2:to_sort/sentiment_experiment.py

biased = get_article_by_name("Monroe College")
biased2 = get_article_by_name("Arun Singh (politician, born 1965)")
neutral = get_article_by_name("Pima Community College")
neutral2 = get_article_by_name("Harvard")

analysis1 = TextBlob(biased)
print(analysis1.sentiment)

analysis2 = TextBlob(biased2)
print(analysis2.sentiment)

analysis3 = TextBlob(neutral)
print(analysis3.sentiment)

analysis4 = TextBlob(neutral2)
print(analysis4.sentiment)



analyzer = SentimentIntensityAnalyzer()
score1 = analyzer.polarity_scores(biased)
print(score1)
score2 = analyzer.polarity_scores(biased2)
print(score2)
score3 = analyzer.polarity_scores(neutral)
print(score3)
score4 = analyzer.polarity_scores(neutral2)
print(score4)