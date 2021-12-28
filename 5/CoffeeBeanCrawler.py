Python 3.9.6 (tags/v3.9.6:db3ff76, Jun 28 2021, 15:26:21) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> from bs4 import BeautifulSoup
>>> import urllib.request
>>> import pandas as pd
>>> import datetime
>>> from selenium import webdriver
>>> import time
>>> def CoffeeBean_store(result):
	CoffeeBean_URL = "https://www.coffeebeankorea.com/store/store.asp"
	wd = webdriver.Chrome("WebDriver/chromedriver.exe")
	for i in range(1, 300):
		wd.get(CoffeeBean_URL)
		time.sleep(1)
		try:
			wd.execute_script("storePop2(%d)" %i)
			time.sleep(1)
			html = wd.page_source
			soupCB = BeautifulSoul(html, )
			]
			
SyntaxError: unmatched ']'
>>> def CoffeeBean_store(result):
	CoffeeBean_URL = "https://www.coffeebeankorea.com/store/store.asp"
	wd = webdriver.Chrome("WebDriver/chromedriver.exe")
	for i in range(1, 300):
		wd.get(CoffeeBean_URL)
		time.sleep(1)
		try:
			wd.execute_script("storePop2(%d)" %i)
			time.sleep(1)
			html = wd.page_source
			soupCB = BeautifulSoul(html, 'html.parser')
			store_name_h2 = soupCB.select("div.store_txt > h2")
			store_name = store_name_h2[0].string
			print(store_name)
			store_info = soupCB.select("div.store_txt > table.store_table > tbody > tr > td")
			store_address_list = list(store_info[2])
			store_address = store_address_list[0]
			store_phone = store_info[3].string
			result.append([store_name]+[store_address]+[store_phone])
		except:
			continue
	return

>>> def main():
	result = []
	print('CoffeeBean store crawling >>>>>>>>>>>>>>>>>>>>>>>>')
	CoffeeBean_store(result)
	CB_tbl = pd.DataFrame(result, columns = ('store', 'address', 'phone'))
	CB_tbl.to_csv('./CoffeeBean.csv', encoding = 'cp949', mode = 'w', index = True)

	
>>> if __name__ == '__main__':
	main()

	
CoffeeBean store crawling >>>>>>>>>>>>>>>>>>>>>>>>
