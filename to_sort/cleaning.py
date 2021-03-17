import re
from nltk.corpus import stopwords

peacock_filename = "peacockterms.txt"
normal_filename = "nonpeacock-ir.txt"

to_process = {peacock_filename, normal_filename}

to_process = {"nonpeacock-ir.txt", "peacockterms.txt"}
stopWords = set(stopwords.words('english'))
for filename in to_process:
    sentences = []
    file = open(filename, 'r')
    for sen in file.readlines():
        # Remove all [footnotes]
        sentence = sen
        sentence = re.sub(r'\[.*?\]', ' ', sentence)

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
    file = open("clean2-" + filename, 'w')
    file.writelines(sentences)

