import requests
from utils.ebay_text_image_parse import ebay_text_image_parse
import time

if __name__ == "__main__":
    ebay_text = "2023-24 Panini Donruss - #188 Giannis Antetokounmpo"
    image_url = "https://i.ebayimg.com/images/g/MQIAAOSwtKdmt0-p/s-l1200.jpg"

    t1 = time.time()
    LLM_output = ebay_text_image_parse(ebay_text=ebay_text, image_url=image_url)
    print('time cost: ', time.time() - t1)
