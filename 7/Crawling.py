from bs4 import BeautifulSoup
import urllib.request
import pandas as pd

result = []

for page in range(1, 29):
    url = 'https://store.steampowered.com/search/?filter=topsellers&specials=1&page='+str(page)
    print('URL : ', url)
    html = urllib.request.urlopen(url)
    soupSteam = BeautifulSoup(html, 'html.parser')
    div_TopSellers = soupSteam.find_all('div', 'responsive_search_name_combined')

    for game in div_TopSellers:
        game_info = game.find_all('span')
        game_title = game_info[0].string
        game_discount = game_info[-2].string
        game_original_price = game_info[-1].string
        #print("Title : ", game_title)
        #print("Discount : ", game_discount)
        #print("Original Price : ", game_original_price)

        game_price = game.find('div','col search_price discounted responsive_secondrow')
        game_price.span.decompose()
        final_price = list(game_price.stripped_strings)
        #print("Final Price : ", final_price[0])

        result.append([game_title]+[' '+game_discount]+[game_original_price]+[final_price[0]])
    print("---------next page---------")

Steam_sale_games = pd.DataFrame(result, columns=('Title', 'Discount', 'Original_Price', 'Price'))
print(Steam_sale_games)
Steam_sale_games.to_csv('./SteamGameSales.csv',encoding='utf-16', mode='w', index=True)
