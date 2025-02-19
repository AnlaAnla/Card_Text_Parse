import requests
import json
import time


api_url = "http://192.168.31.146:9000/parse_ebay_data"
ebay_text = "2023-24 Panini Mosaic Prizm Bank Shot #10 Kevin Durant Phoenix Suns"

data = {"ebay_text": ebay_text}

t1 = time.time()
response = requests.post(api_url, data=json.dumps(data))
print(response.json())
print('time cost: ', time.time()-t1)
