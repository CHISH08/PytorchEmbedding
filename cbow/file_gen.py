import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

text = ""
for i in tqdm(range(1, 362)):
    url = f'https://ilibrary.ru/text/11/p.{i}/index.html'

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    text += soup.get_text()
f = open('file.txt', 'w')

f.write(text)

f.close()