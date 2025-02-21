import requests

def text_vecSearch(url, text, top_k=5):
    # url = "http://127.0.0.1:9002/search_cardSet"
    data = {"text": text, "topk": top_k}
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        # print(response.json())
        result_list = []
        data = response.json()['results']
        for item in data:
            result_list.append(item['name'])

        return result_list
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return ''
