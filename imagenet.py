#image_netからランダムに100枚とってきて
#フォルダに入れておく

import pandas as pd
from urllib import request
from numpy.random import *

IMG_LIST_URL="http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid={}"

n = 100
ransu = randint(0,82115, n) #ncol=82115 
words = pd.read_csv("/home/zhaoyin-t/words.txt", header=None, delimiter="\t")

for i in range(n):
	tmp = words.iloc[ransu[i], 0]
	url = IMG_LIST_URL.format(tmp)
	with request.urlopen(url) as response:
		try:
			html = response.read()  	
			data = html.decode()
			data = data.split()
			urls = data[1::2]
			print(urls)
		except:
			continue

