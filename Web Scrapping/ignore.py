import requests
from bs4 import BeautifulSoup
import csv

# Send a GET request to the URL
url = 'http://127.0.0.1:5500/jskjdkas.html' # Enter webpage url
response = requests.get(url)

soup = BeautifulSoup(response.text, 'html.parser')

# Find all product cards
cards = soup.find_all('div', class_='card')

for i in range (len(cards)): #only print titles in terminal
    print(f"Product {i+1} {cards[i].find('div', class_='title').text.strip()}")
