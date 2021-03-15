"""
leave mark in for citation
quotes?
"""

import re

peacock_filename = "peacockterms.txt"
normal_filename = "nonpeacock-ir.txt"

to_process = {peacock_filename, normal_filename}

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
        print(sentence)
        sentences.append(sentence)
    file.close()
    file = open("NEW-clean-" + filename, 'w')
    file.writelines(sentences)

