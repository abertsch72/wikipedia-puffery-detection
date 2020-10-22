import requests
import wikipedia
import sys


print(sys.argv[1])
text = requests.get(sys.argv[1]).json()


tentative = set()
print(text)
for pagename in text['query']['categorymembers']:
    try:
        print(pagename)
        t = wikipedia.page(pagename['title'])
        raw = t.html().split('\n')[4:]
        for line in raw:
            if "peacock" in line:
                tentative.add(pagename['title'])
                print(pagename['title'])
    except:
        pass
print(tentative)
print(len(tentative))