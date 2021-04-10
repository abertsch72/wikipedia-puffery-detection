import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

peacock_filename = "peacockterms.txt"
normal_filename = "nonpeacock_random_append.txt"

to_process = {peacock_filename, normal_filename}


for filename in to_process:
    sentences = []
    short_sentences = []
    stop_free = []
    stop_short = []
    file = open(filename, 'r')
    for sen in file.read().split("~~~~"):
        # Remove all [footnotes]
        sentence = BeautifulSoup(sen, "html.parser").text
        sentence = re.sub(r'\[.*?\]', ' ', sentence)

        # Remove all the special characters
        sentence = re.sub(r'\W', ' ', sentence)

        # remove all single characters
        sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)

        # Remove single characters from the start
        sentence = re.sub(r'\^[a-zA-Z]\s+', ' ', sentence)

        sentence = re.sub(r'\^.*learn how and when to remove this message', ' ', sentence)
        sentence = sentence.split("template message")[-1]

        # Substituting multiple spaces with single space
        sentence = re.sub(r'\s+', ' ', sentence, flags=re.I)

        # Removing prefixed 'b'
        sentence = re.sub(r'^b\s+', '', sentence)

        sentence = sentence.strip()

        # Converting to Lowercase
        sentence = sentence.lower() + "\n"
        print(sentence)
        sentences.append(sentence)
        short_sentences.append(' '.join(sentence.split()[0:400]) + "\n")
        if(sentence != None):
            word_tokens = word_tokenize(sentence)

            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            stop_free.append(' '.join(filtered_sentence) + "\n")
        if sentence != None:
            word_tokens = word_tokenize(sentence)

            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            stop_short.append(' '.join(filtered_sentence[0:400]) + "\n")
    file.close()
    file = open("clean-" + filename, 'w')
    file2 = open("clean-short-" + filename, 'w')
    file3 = open("clean-nostop-" + filename, 'w')
    file4 = open("clean-short-nostop-" + filename, 'w')
    file.writelines(sentences)
    file2.writelines(short_sentences)
    file3.writelines(stop_free)
    file4.writelines(stop_short)

