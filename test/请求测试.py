import requests
import json
import time


# api_url = "http://192.168.31.146:9000/parse_ebay_data_LLM"
api_url = "http://127.0.0.1:9000/parse_ebay_data"

# Stephen Curry 2021-22 Select Blue Shimmer Prizm Golden State Warriors C8
ebay_text = "Jalen Hood-Schifino 2023-24 Panini Mosaic Silver Mosaic RC Los Angeles Lakers"

data = {"ebay_text": ebay_text}

t1 = time.time()
response = requests.post(api_url, data=json.dumps(data))
print(response.json())
print('time cost: ', time.time()-t1)
