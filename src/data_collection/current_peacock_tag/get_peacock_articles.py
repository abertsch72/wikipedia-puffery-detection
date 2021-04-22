"""
look at regex instead of re
figure out what's going on with last 4 keywords
writelines
tighten up exception block
"""

import requests
import wikipedia
from bs4 import BeautifulSoup
import re
from string import punctuation, whitespace
from nltk import tokenize

cmd= "https://en.wikipedia.org/w/api.php?action=query&format=json&list=categorymembers&cmtitle=Category%3A%20All%20articles%20with%20peacock%20terms&cmlimit=max"
text = requests.get(cmd).json()


regex = re.compile(r"(.*?)(<[^<]*>)*<[^<]*?(Puffery|peacock)[^<]*?>") # match a sentence with puffery/peacock tag

peacock = ""
nonpeacock = ""

pages_used = set()
cont = True
while(cont):
    print(text.get('continue'))
    for pagename in text['query']['categorymembers']:
        try:
            t = wikipedia.page(pagename['title'])
            raw = t.html().split('\n')[4:] # remove very top of the article
            for line in raw:
                if "peacock" in line.lower():
                    line = line.lstrip(whitespace)
                    if not line.startswith("<table"): # discard anything at the start of a table (usually appears near the top, not in a content section)
                        m = regex.match(line)
                        if m:
                            keywords = BeautifulSoup(m.group(0), "html.parser").textformatting
                            keywords = keywords.split()
                            keywords = ' '.join(keywords[-4:]).split('.')[0].strip(punctuation) # why ?
                            sentences = tokenize.sent_tokenize(BeautifulSoup(line, "html.parser").text)
                            potential = [sent for sent in sentences if keywords in sent][0]
                            if "contains wording that promotes the subject in a subjective manner without imparting real information" in potential:
                                # this is a section header, not a sentence tag
                                continue
                            else:
                                print(potential)
                                peacock += potential + "\n"
                            i = sentences.index(potential)
                            if i > 0:
                                # add the sentence before this one to nonpeacock
                                print(sentences[sentences.index(potential)-1])
                                nonpeacock += sentences[sentences.index(potential)-1]
                            if i < len(sentences) - 1:
                                # add the sentence after this one to nonpeacock
                                print(sentences[sentences.index(potential)+1])
                                nonpeacock += sentences[sentences.index(potential)+1]
                            pages_used.add(pagename['title'])
        except Exception as e:
            print(e)

    c = text.get("continue")
    print(c)
    if c is not None:
        text = requests.get(cmd + "&cmcontinue=" + c.get("cmcontinue")).json()
    else:
        f = open("../../../data/peacockterms.txt", 'w')
        f.write(peacock)
        f1 = open("../../../data/nonpeacockterms.txt", 'w')
        f1.write(nonpeacock)
        import sys
        sys.exit(0)
