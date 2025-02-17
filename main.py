import requests
from utils.ebay_text_image_parse import ebay_text_image_parse
import time

if __name__ == "__main__":
    ebay_text = "2023-24 Panini Court Kings Colby Jones RC -Fresh Paint Red  Auto /99 Kings"
    image_url = "https://i.ebayimg.com/images/g/6x4AAOSwBEhm3Gi0/s-l1200.jpg"

    t1 = time.time()
    LLM_output = ebay_text_image_parse(ebay_text=ebay_text, image_url=image_url)
    print('time cost: ', time.time() - t1)
