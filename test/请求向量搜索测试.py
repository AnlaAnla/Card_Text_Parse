from utils.program_cardSet_vecSearch import text_vecSearch
import time
import json

Config_path = "../Config.json"
Config = json.load(open(Config_path))

VEC_SEARCH_PROGRAM_API_URL = Config['VEC_SEARCH_PROGRAM_API_URL']
VEC_SEARCH_CARD_SET_API_URL = Config['VEC_SEARCH_CARD_SET_API_URL']


ebay_text = " Purple Parallel  Optic Basketball Los Angeles Lakers"

data = {"ebay_text": ebay_text}

t1 = time.time()
result = text_vecSearch(VEC_SEARCH_CARD_SET_API_URL, ebay_text, top_k=10)
print(result)
print('time cost: ', time.time()-t1)
