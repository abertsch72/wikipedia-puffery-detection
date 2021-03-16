"""
There's a special wikipedia dump for all titles in namespace 0 (approx. 4.5M)
can get last 500 edits for each page using pywikibot

with content=True: 604.7476875782013
with content=False: 44.73729753494263

I looked into getting the revisions through the API but it does not appear that pywikibot is really the slowdown here--
it's just that some pages are HUGE so revisions are expensive.

"""
import time
import pywikibot
from joblib import Parallel, delayed
import multiprocessing
from nltk import tokenize
from bs4 import BeautifulSoup
import re

regex = re.compile(r"{{.*?}}")
names = []
start = 0

def get_peacock(name):
    global regex

    site = pywikibot.Site("en", "wikipedia")
    page = pywikibot.Page(site, name)
    start = time.time()
    revs = page.revisions(content=True)
    end = time.time()
    curr = next(revs).get("slots").get("main").get("*")
    sample = []
    for line in tokenize.sent_tokenize(BeautifulSoup(re.sub(r"{{.*?}}|===.*?===|[\[Category:.*?]\]", "", page.text), "html.parser").text):
        line = line.split("\n")[0]
        if line != '\n' and '.' in line:
            sample.append(line)
    if "{{Multiple issues" in curr or "date=" in curr:
        #print("THROW OUT")
        sample = []
    else:
        #print("KEEP")
        from random import shuffle
        shuffle(sample)

    notFound = True

    peacock = []
    while notFound:
        prev = ""
        for line in tokenize.sent_tokenize(
                BeautifulSoup(page.text, "html.parser").text):
            if "{peacock" in line or "{Puffery" in line:
                if prev.endswith("."):
                    peacock.append(prev)
                    print(prev)
                else:
                    peacock.append(line)
                    print(line)
                notFound = False
            prev = line
        curr = next(revs, -1)
        if curr == -1:
            #print("END, NOT FOUND")
            break
        else:
            curr = curr.get("slots").get("main").get("*")
    sample = sample[:min(len(sample), len(peacock))]
    return peacock, sample

def run_job(i):
    global names, start
    print("number: " + str(i + start) + "\tname: " + names[i])
    val = ([], [])
    try:
        val = get_peacock(names[i])
    except:
        pass

    return val

def parallel_work():
    global names
    num_cores = multiprocessing.cpu_count() - 2
    inputs = range(len(names))
    with multiprocessing.Pool(num_cores) as p:
        results = [t for t in p.map(run_job, inputs) if t != ([], [])]
    #results = Parallel(n_jobs=2)(delayed(run_job)(i) for i in inputs)

    print("results:")
    print(results)
    import pickle
    pickle.dump(results, open("results5.pkl", 'wb'))

filename = "enwiki-latest-all-titles-in-ns0"
num_cores = multiprocessing.cpu_count()
num = 10000
start = 430000
with open(filename) as f:
    f.readline() # the header
    for i in range(start):
        f.readline()
    for i in range(num):
        names.append(f.readline().strip("\n"))
print(names)
parallel_work()
#print(get_peacock("Honda CL77"))


"""# throw out
print("throw out these:")
get_revisions("Fanfani II Cabinet")
get_revisions("Domingo Terán de los Ríos")
get_revisions("Road cycling")
get_revisions("Qafë Vranicë")
get_revisions("Tempo (company)")
get_revisions("Álvaro Ortiz")
get_revisions("Trendspotting magazine")

get_revisions("Yokokawa Dam")

# keep
print("keep these:")
get_revisions("Rithvik Dhanjani")
get_revisions("Ste. Rose du Lac Airport")
get_revisions("Princess Anna Sophie of Denmark")
get_revisions("Hong Kong Dragon Boat Festival in New York")
get_revisions("Jesús Rojas (Venezuelan boxer)")
get_revisions("Hoseynabad, Paskhan")
"""