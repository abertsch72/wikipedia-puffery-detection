<<<<<<< HEAD:to_sort/src/initial_models/sentiment_threshold.py
from src.data_collection.current_peacock_tag.get_article_by_name import get_article_by_name
=======
from to_sort.get_article_by_name import get_article_by_name
>>>>>>> a96341abddd6e09186592d5da09defe66b974bb2:to_sort/sentiment_threshold.py
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


articles = [
    get_article_by_name("Arcon2"),
    get_article_by_name("All India Students Federation"),
    get_article_by_name("Ashoka University"),
    get_article_by_name("Jeff Gillette"),
    get_article_by_name("Han Chiang University College of Communication"),
    get_article_by_name("Mostafa Heravi")
]

good_articles = [
    get_article_by_name("Hamamelis vernalis"),
    get_article_by_name("Mathematical Association of America"),
    get_article_by_name("John Stuart Williams"),
    get_article_by_name("Arif Dirlik"),
    get_article_by_name("Johnny \"Hammond\" Cooks with Gator Tail"),
    get_article_by_name("Phir Milenge (1942 film)")
]

textblob_sentiment = []
vader_sentiment = []

analyzer = SentimentIntensityAnalyzer()

for article in articles:
    textblob_sentiment.append(TextBlob(article).sentiment)
    vader_sentiment.append(analyzer.polarity_scores(article))

for article in good_articles:
    textblob_sentiment.append(TextBlob(article).sentiment)
    vader_sentiment.append(analyzer.polarity_scores(article))

print(textblob_sentiment[0:6])
print(vader_sentiment[0:6])

print(textblob_sentiment[6:])
print(vader_sentiment[6:])

"""

    Hill House, Helensburgh

M

    Minimalism

P

    Manny Puig

R

    Rick Hansen Secondary School (Mississauga)

S

    Shaka Bundu
    Dolores Sison

T

    Mario Telò"""