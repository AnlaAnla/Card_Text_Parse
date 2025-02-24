import requests
import json
import time


# api_url = "http://192.168.31.146:9000/parse_ebay_data_LLM"
api_url = "http://127.0.0.1:9000/parse_ebay_data"

# Brandon Miller 2023-24 Panini Origins Basketball Euphoria RC #5
# Cameron Johnson 2023-24 Haunted Hoops Basketball Candy Corn Card # 93

# 2023-24 Panini Select - Concourse White Disco Kristaps Porzingis 49/75 Szs
# 2023-24 Panini Mosaic - GG Jackson II - Green Mosaic Prizm RC #228 SP Grizzlies
ebay_text = "2023 PANINI SELECT ORANGE FLASH #80 BRANDON MILLER ROOKIE RC PSA 7"

# 这是第

data = {"ebay_text": ebay_text}

t1 = time.time()
response = requests.post(api_url, data=json.dumps(data))
print(response.json())
print('time cost: ', time.time()-t1)
