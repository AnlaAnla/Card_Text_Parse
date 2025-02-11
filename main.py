import requests
import json
from PIL import Image
import io
import pandas as pd
import re
from utils.call_predict_with_image import call_predict_with_image
from utils.call_predict_with_image2 import call_predict_with_image2
from utils.program_cardSet_vecSearch import text_vecSearch


def preprocess_year_num(text):
    year_match = re.search(r'\b(20\d{2})(?:-\d{2})?\b', text)
    num_match = re.search(r'#([A-Za-z0-9\-]+)', text)
    return {
        'year': year_match.group(1) if year_match else "",
        'card_num': num_match.group(1) if num_match else ""
    }


def match_tag(csv_data, tag, text):
    # 和数据库匹配
    matches = csv_data[csv_data[tag].str.contains(f"^{text}$", na=False, regex=True, case=False)]
    return not matches.empty


# LLM 服务器的地址 (根据你的实际情况修改)
LLM_API_URL = "http://127.0.0.1:9001/predict_with_image"
VEC_SEARCH_PROGRAM_API_URL = "http://127.0.0.1:9002/search_program"
VEC_SEARCH_CARD_SET_API_URL = "http://127.0.0.1:9002/search_cardSet"

if __name__ == "__main__":

    question_with_image = "2022 Prestige Living Legends Xtra Points Blue Dwight Freeney /299 numbered"
    image_url = "https://i.ebayimg.com/images/g/~IkAAOSw53FmfNEl/s-l1200.jpg"

    program_cardSet_athlete_data = pd.read_csv("Data\program_cardSet_athlete.csv")
    brand_data_list = ["Playoff", "Donruss", "Score", "Panini"]

    # 1 获取年份和编号
    preprocess_year_num_data = preprocess_year_num(question_with_image)
    print('1 获取年份和编号: ', preprocess_year_num_data['year'], preprocess_year_num_data['card_num'])

    # 2 获取球员名称
    LLM_output = call_predict_with_image(LLM_API_URL, image_url, question_with_image)
    predict_athlete = LLM_output['athlete']

    # 和数据库匹配
    if match_tag(program_cardSet_athlete_data, tag="athlete", text=predict_athlete):
        print('2 获取球员名称: ', predict_athlete)
    else:
        predict_athlete = ''

    # 3 第一次验证program和card_set
    predict_program = LLM_output['program']
    predict_cardSet = LLM_output['card_set']
    if match_tag(program_cardSet_athlete_data, tag='program', text=predict_program):
        print('3 获取 program: ', predict_program)
    else:
        predict_program = ''

    if match_tag(program_cardSet_athlete_data, tag='card_set', text=predict_cardSet):
        print("3 获取 card_set: ", predict_cardSet)
    else:
        predict_cardSet = ''

    # 4 向量搜索 program 和 card_set
    print('_' * 20)
    vec_text = question_with_image
    # 去掉厂商
    for brand in brand_data_list:
        if brand in vec_text:
            vec_text = vec_text.replace(brand, '')
    if preprocess_year_num_data['year'] != '':
        vec_text = vec_text.replace(preprocess_year_num_data['year'], '')
    if preprocess_year_num_data['card_num'] != '':
        vec_text = vec_text.replace(preprocess_year_num_data['card_num'], '')
        vec_text = vec_text.replace('#', '')
    if predict_athlete != '':
        vec_text = vec_text.replace(predict_athlete, '')

    print('vec_text: ', vec_text)
    top_k_programs = text_vecSearch(VEC_SEARCH_PROGRAM_API_URL, vec_text)
    top_k_card_sets = text_vecSearch(VEC_SEARCH_CARD_SET_API_URL, vec_text)
    print(top_k_programs)
    print(top_k_card_sets)
    print('_' * 20)

    # 5 第二次LLM提取
    LLM_output2 = call_predict_with_image2(LLM_API_URL, image_path=image_url, ebay_text=vec_text,
                                           year=preprocess_year_num_data['year'],
                                           card_num=preprocess_year_num_data['card_num'],
                                           athlete=predict_athlete,
                                           top_k_programs=top_k_programs,
                                           top_k_card_sets=top_k_card_sets)

    if predict_program == '' and match_tag(program_cardSet_athlete_data, tag='program', text=LLM_output2['program']):
        predict_program = LLM_output2['program']
        print('4 获取 program: ', predict_program)
    if match_tag(program_cardSet_athlete_data, tag='card_set', text=LLM_output2['card_set']):

        # card_set 严格判断
        set_cardSet_flag = False
        for word in LLM_output2['card_set'].split(' '):
            if word in question_with_image or word.lower() == 'base':
                set_cardSet_flag = True
                pass
            else:
                set_cardSet_flag = False
                break
        if set_cardSet_flag:
            predict_cardSet = LLM_output2['card_set']
            print("4 获取 card_set: ", predict_cardSet)

    print('_' * 20)
    print(question_with_image)
    print("LLM : ", LLM_output)
    print('LLM2:', LLM_output2)

    LLM_output['year'] = preprocess_year_num_data['year']
    LLM_output['card_num'] = preprocess_year_num_data['card_num']
    LLM_output['athlete'] = predict_athlete

    LLM_output['program'] = predict_program
    LLM_output['card_set'] = predict_cardSet

    print('修订结果: ', LLM_output)
