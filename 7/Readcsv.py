from bs4 import BeautifulSoup
import urllib.request

url = 'https://store.steampowered.com/search/?filter=topsellers&specials=1&page=1'
html = urllib.request.urlopen(url)
soupSteam = BeautifulSoup(html, 'html.parser')

print(soupSteam.prettify())