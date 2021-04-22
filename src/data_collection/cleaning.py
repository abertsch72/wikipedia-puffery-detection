"""
should this do something about quotes?
"""

import re
from nltk.corpus import stopwords

to_process = {"nonpeacock-ir.txt", "peacockterms.txt"}
stopWords = set(stopwords.words('english'))
for filename in to_process:
    sentences = []
    with open(filename, 'r') as file:
        for sen in file.readlines():
            sentence = sen

            # Replace all [citations] with a keyword (CITE) to signal citation
            # note that citations are numbers in [square brackets]
            # this is distinct from non-citation [footnotes], which are words in square brackets
            sentence = re.sub(r'(?<!^)\[[0-9]+\]', 'CITE', sentence)

            # Remove all non-citation [footnotes]
            sentence = re.sub(r'(?!\[\])\[.+?\]', ' ', sentence)

            # Remove all the special characters
            sentence = re.sub(r'\W', ' ', sentence)

            # remove all single characters
            sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)

            # Remove single characters from the start -- \w with regex
            sentence = re.sub(r'\^[a-zA-Z]\s+', ' ', sentence)

            # Substituting multiple spaces with single space
            sentence = re.sub(r'\s+', ' ', sentence, flags=re.I)

            # Removing prefixed 'b'
            sentence = re.sub(r'^b\s+', '', sentence)

            sentence = sentence.strip()

            # Converting to lowercase
            sentence = sentence.lower() + "\n"

            # Removing stopwords
            sentence = " ".join([word for word in sentence.split() if word not in stopWords])

            sentence += "\n"
            print(sentence)
            sentences.append(sentence)

    with open("clean-" + filename, 'w') as file:
        file.writelines(sentences)


