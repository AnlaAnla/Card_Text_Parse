import requests
from utils.ebay_text_image_parse import ebay_text_image_parse
import time

if __name__ == "__main__":
    ebay_text = "2021-22 Chrome Overtime Elite Levitate #LEV-12 Tyler Smith / - Overtime Elite"
    # image_url = "https://i.ebayimg.com/images/g/WdUAAOSw2jhmy8s7/s-l1200.jpg"
    image_url = "Data/temp.jpg"

    t1 = time.time()
    LLM_output = ebay_text_image_parse(ebay_text=ebay_text, image_url=image_url)
    print('time cost: ', time.time() - t1)
