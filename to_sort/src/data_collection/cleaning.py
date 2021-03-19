"""
leave mark in for citation
quotes?
"""

import re
from nltk.corpus import stopwords

<<<<<<< HEAD:to_sort/src/data_collection/cleaning.py
peacock_filename = "../../data/peacockterms.txt"
normal_filename = "../../data/nonpeacock-ir.txt"
=======
peacock_filename = "peacockterms.txt"
normal_filename = "nonpeacock-ir.txt"
>>>>>>> a96341abddd6e09186592d5da09defe66b974bb2:to_sort/cleaning.py

to_process = {peacock_filename, normal_filename}

to_process = {"nonpeacock-ir.txt", "peacockterms.txt"}
stopWords = set(stopwords.words('english'))
for filename in to_process:
    sentences = []
    file = open(filename, 'r')
    for sen in file.readlines():
        sentence = sen

        # Replace all [citations] with an empty [] to signal citation
        # note that citations are numbers in [square brackets]
        # should this be a cite keyword instead??
        sentence = re.sub(r'(?<!^)\[[0-9]+\]', 'CITE', sentence)

        # Remove all non-citation [footnotes]
        sentence = re.sub(r'(?!\[\])\[.+?\]', ' ', sentence)

        # Remove all the special characters
        sentence = re.sub(r'\W', ' ', sentence)

        # remove all single characters
        sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)

        # Remove single characters from the start
        sentence = re.sub(r'\^[a-zA-Z]\s+', ' ', sentence)

        # Substituting multiple spaces with single space
        sentence = re.sub(r'\s+', ' ', sentence, flags=re.I)

        # Removing prefixed 'b'
        sentence = re.sub(r'^b\s+', '', sentence)

        sentence = sentence.strip()

        # Converting to Lowercase
        sentence = sentence.lower() + "\n"
        sentence = " ".join([word for word in sentence.split() if word not in stopWords])
        sentence += "\n"
        print(sentence)
        sentences.append(sentence)
    file.close()
<<<<<<< HEAD:to_sort/src/data_collection/cleaning.py
    file = open("NEW-clean-" + filename, 'w')
=======
    file = open("clean2-" + filename, 'w')
>>>>>>> a96341abddd6e09186592d5da09defe66b974bb2:to_sort/cleaning.py
    file.writelines(sentences)

