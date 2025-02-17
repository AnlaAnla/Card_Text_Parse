import requests
import json


api_url = "http://172.24.144.1:9000/parse_ebay_data"
ebay_text = "2021-22 Panini Prizm Red Ice Prizm Tim Duncan #268 HOF"
image_url = "https://i.ebayimg.com/images/g/WdUAAOSw2jhmy8s7/s-l1200.jpg"

data = {"ebay_text": ebay_text, "image_url": image_url}


response = requests.post(api_url, data=json.dumps(data))
print(response.json())
