import xml.etree.ElementTree as ET

tree = ET.parse('/home/abertsch/Downloads/wiki.xml')
root = tree.getroot()

for child in root:
    for ch2 in child:
        for ch3 in ch2:
            for ch4 in ch3:
                print(ch4.text)
                for ch5 in ch4:
                    print(ch5.text)


