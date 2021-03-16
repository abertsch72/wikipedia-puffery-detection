import wikipedia

def get_article_by_name(name):
    page = wikipedia.page(name)
    return page.content

#get_article_by_name("Monroe College")
#get_article_by_name("Category:Articles with peacock terms from December 2019")