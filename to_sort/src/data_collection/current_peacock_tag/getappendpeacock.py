import requests
import wikipedia
from bs4 import BeautifulSoup
import re
from string import punctuation, whitespace
from nltk import tokenize

cmd= "https://en.wikipedia.org/w/api.php?action=query&format=json&list=categorymembers&cmtitle=Category%3A%20All%20articles%20with%20peacock%20terms&cmlimit=max"
text = requests.get(cmd).json()


regex = re.compile(r"(.*?)(<[^<]*>)*<[^<]*?(Puffery|peacock)[^<]*?>")

peacock = ""
nonpeacock = ""


f = open("../../../data/appendpeacock.txt", 'w')
f1 = open("../../../data/appendnormal.txt", 'w')

pages_used = set()
cont = True
while(cont):
    print(text.get('continue'))
    for pagename in text['query']['categorymembers']:
        try:

            t = wikipedia.page(pagename['title'])
            raw = t.html().split('\n')[4:]
            for line in raw:
                if "peacock" in line.lower():
                    line.lstrip(whitespace)
                    if (line[0:6] != "<table"):
                        m = regex.match(line)
                        if m:
                            keywords = BeautifulSoup(m.group(0), "html.parser").text
                            keywords = keywords.split()
                            keywords = ' '.join(keywords[max(0, len(keywords)-4):]).split('.')[0].strip(punctuation)
                            sentences = tokenize.sent_tokenize(BeautifulSoup(line, "html.parser").text)
                            potential = [sent for sent in sentences if keywords in sent][0]
                            if "contains wording that promotes the subject in a subjective manner without imparting real information" in potential:
                                continue
                            else:
                                print(potential)
                                f.write(potential + "\n")
                                peacock += potential + "\n"
                            i = sentences.index(potential)
                            if i > 0:
                                print(sentences[sentences.index(potential)-1])
                                nonpeacock += sentences[sentences.index(potential)-1] + "\n"
                                f1.write(sentences[sentences.index(potential)-1] + "\n")
                            if i < len(sentences) - 1:
                                print(sentences[sentences.index(potential)+1])
                                nonpeacock += sentences[sentences.index(potential)+1] + "\n"
                                f1.write(sentences[sentences.index(potential)+1] + "\n")
                            pages_used.add(pagename['title'])
        except Exception as e:
            print(e)
            print(t)
    c = text.get("continue")
    print(c)
    if c is not None:
        text = requests.get(cmd + "&cmcontinue=" + c.get("cmcontinue")).json()
    else:
        import sys
        sys.exit(0)

print(pages_used)
print(nonpeacock)
print("=================================")
print(peacock)
