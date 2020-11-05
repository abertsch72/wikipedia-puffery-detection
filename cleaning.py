import re

peacock_filename = "peacockterms.txt"
normal_filename = "nonpeacockterms.txt"

to_process = {peacock_filename, normal_filename}

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
        print(sentence)
        sentences.append(sentence)
    file.close()
    file = open("clean-" + filename, 'w')
    file.writelines(sentences)

