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


def judge_by_vec_search_list(ebay_text: str, vector_list: list[str], pass_word_list: list[str] = None):
    # 用向量搜索的结果对比原文本
    '''
    :param ebay_text:
    :param vector_list:
    :param pass_word_list: 搜索文本中忽略这些单词的匹配
    :return:
    '''
    # 根据单词数量从多到少排序
    vector_list = sorted(vector_list, key=lambda s: len(s.split()), reverse=True)

    for i in range(len(pass_word_list)):
        # 全部小写
        pass_word_list[i] = pass_word_list[i].lower()

    for program in vector_list:
        set_program_flat = False
        for word in program.split(' '):
            if word in pass_word_list or word == '':
                continue

            if word.lower() in ebay_text.lower().split(' '):
                set_program_flat = True
            else:
                set_program_flat = False
                break
        if set_program_flat:
            return program
    return False


def get_vec_search_judge_result(vec_search_url: str, ebay_text: str, vec_text: str, pass_word_list: list[str] = None):
    top_k_list = text_vecSearch(vec_search_url, vec_text, top_k=10)
    print(top_k_list)
    # program 严格判断
    judge_result = judge_by_vec_search_list(ebay_text=ebay_text,
                                            vector_list=top_k_list,
                                            pass_word_list=pass_word_list)
    # pass_word_list=['the', 'and'])
    if judge_result is not False:
        return judge_result
        # predict_result = judge_result
        # print("4 向量搜索后获得 program: ", predict_result)
    return ''


def judge_cardSet(ebay_text: str, card_set: str):
    # card_set 严格判断
    set_cardSet_flag = False
    for word in card_set.split(' '):
        if word.lower() in ebay_text.lower().split(' ') or word.lower() == 'base':
            set_cardSet_flag = True
        else:
            set_cardSet_flag = False
            break
    if set_cardSet_flag:
        return True
    return False


def ebay_text_image_parse(ebay_text, image_url):
    program_pass_word_list = ['the', 'and']
    cardSet_pass_word_list = ['base']

    program_cardSet_athlete_data = pd.read_csv("Data\program_cardSet_athlete.csv")

    # 1 获取年份和编号
    preprocess_year_num_data = preprocess_year_num(ebay_text)
    print('1 获取年份和编号: ', preprocess_year_num_data['year'], preprocess_year_num_data['card_num'])

    # 2 LLM获取球员名称
    LLM_output = call_predict_with_image(LLM_API_URL, image_url, ebay_text)
    predict_athlete = LLM_output['athlete']

    # 和数据库匹配
    if match_tag(program_cardSet_athlete_data, tag="athlete", text=predict_athlete):
        print('2 获取球员名称: ', predict_athlete)
    else:
        predict_athlete = ''

    # 以下三个tag结束
    LLM_output['year'] = preprocess_year_num_data['year']
    LLM_output['card_num'] = preprocess_year_num_data['card_num']
    LLM_output['athlete'] = predict_athlete

    # 3 第一次验证program和card_set
    predict_program = LLM_output['program']
    predict_cardSet = LLM_output['card_set']

    if predict_program != '':
        predict_program_flag = False
        if match_tag(program_cardSet_athlete_data, tag='program', text=predict_program):
            print('3 获取 program: ', predict_program)
            predict_program_flag = True
        if not predict_program_flag:
            # 精确匹配后, 用向量匹配重新匹配
            predict_program = get_vec_search_judge_result(vec_search_url=VEC_SEARCH_PROGRAM_API_URL,
                                                          ebay_text=ebay_text,
                                                          vec_text=predict_program,
                                                          pass_word_list=program_pass_word_list)
            predict_program_flag = True
        if not predict_program_flag:
            predict_program = ''
        del predict_program_flag

    if predict_cardSet != '':
        predict_cardSet_flag = False
        if match_tag(program_cardSet_athlete_data, tag='card_set', text=predict_cardSet):
            print("3 获取 card_set: ", predict_cardSet)
            predict_cardSet_flag = True
        if not predict_cardSet_flag:
            predict_cardSet = get_vec_search_judge_result(vec_search_url=VEC_SEARCH_CARD_SET_API_URL,
                                                          ebay_text=ebay_text,
                                                          vec_text=predict_cardSet,
                                                          pass_word_list=cardSet_pass_word_list)
            predict_cardSet_flag = True
        if not predict_cardSet_flag:
            predict_cardSet = ''
        del predict_cardSet_flag

    if predict_program != '' and predict_cardSet != '':
        LLM_output['program'] = predict_program
        LLM_output['card_set'] = predict_cardSet
        print('++++结果++++: ', LLM_output)
        return LLM_output

    # 4 向量搜索 program , card_set
    print('_' * 20)
    vec_text = ebay_text
    if preprocess_year_num_data['year'] != '':
        vec_text = vec_text.replace(preprocess_year_num_data['year'], '')
    if preprocess_year_num_data['card_num'] != '':
        vec_text = vec_text.replace(preprocess_year_num_data['card_num'], '')
        vec_text = vec_text.replace('#', '')
    if predict_athlete != '':
        vec_text = vec_text.replace(predict_athlete, '')

    # 在这里根据 program 和 card_set 的有无分为三种情况
    if predict_program == '' and predict_cardSet == '':
        print('vec_text: ', vec_text)
        predict_program = get_vec_search_judge_result(vec_search_url=VEC_SEARCH_PROGRAM_API_URL,
                                                      ebay_text=ebay_text,
                                                      vec_text=vec_text,
                                                      pass_word_list=program_pass_word_list)

        # 如果存在 program 那么从文本里去除 program
        if predict_program != '':
            vec_text.replace(predict_program, '')
            print('vec_text [去除program]: ', vec_text)

        predict_cardSet = get_vec_search_judge_result(vec_search_url=VEC_SEARCH_CARD_SET_API_URL,
                                                      ebay_text=ebay_text,
                                                      vec_text=vec_text,
                                                      pass_word_list=cardSet_pass_word_list)

    elif predict_program == '' and predict_cardSet != '':
        vec_text.replace(predict_cardSet, '')
        print('vec_text [去除cardSet]: ', vec_text)
        predict_program = get_vec_search_judge_result(vec_search_url=VEC_SEARCH_PROGRAM_API_URL,
                                                      ebay_text=ebay_text,
                                                      vec_text=vec_text,
                                                      pass_word_list=program_pass_word_list)
    else:
        vec_text.replace(predict_program, '')
        print('vec_text [去除program]: ', vec_text)
        predict_cardSet = get_vec_search_judge_result(vec_search_url=VEC_SEARCH_CARD_SET_API_URL,
                                                      ebay_text=ebay_text,
                                                      vec_text=vec_text,
                                                      pass_word_list=cardSet_pass_word_list)

    # 5 第二次LLM提取
    # LLM_output2 = call_predict_with_image2(LLM_API_URL, image_path=image_url, ebay_text=vec_text,
    #                                        year=preprocess_year_num_data['year'],
    #                                        card_num=preprocess_year_num_data['card_num'],
    #                                        athlete=predict_athlete,
    #                                        top_k_programs=top_k_programs,
    #                                        top_k_card_sets=top_k_card_sets)
    #
    # try:
    #     if predict_program == '' and match_tag(program_cardSet_athlete_data, tag='program',
    #                                            text=LLM_output2['program']):
    #         predict_program = LLM_output2['program']
    #         print('5 获取 program: ', predict_program)
    # except Exception as e:
    #     print("LLM_output2 二次检验 program 异常, 跳过!!")
    #
    # try:
    #     if predict_cardSet == '' and match_tag(program_cardSet_athlete_data, tag='card_set', text=LLM_output2['card_set']):
    #         if judge_cardSet(ebay_text=question_with_image, card_set=LLM_output2['card_set']):
    #             predict_cardSet = LLM_output2['card_set']
    #             print("4 获取 card_set: ", predict_cardSet)
    # except Exception as e:
    #     print("LLM_output2 二次检验 card_set异常, 跳过!!")

    print('_' * 20)
    print(ebay_text)
    print("LLM : ", LLM_output)
    # print('LLM2:', LLM_output2)

    LLM_output['program'] = predict_program
    LLM_output['card_set'] = predict_cardSet

    print('++++结果++++: ', LLM_output)

    return LLM_output


# LLM 服务器的地址 (根据你的实际情况修改)
LLM_API_URL = "http://127.0.0.1:9001/predict_with_image"
VEC_SEARCH_PROGRAM_API_URL = "http://127.0.0.1:9002/search_program"
VEC_SEARCH_CARD_SET_API_URL = "http://127.0.0.1:9002/search_cardSet"

if __name__ == "__main__":
    ebay_text = "2021 Upper Deck Goodwin Champions Memorabilia Mac Jones #M-JO RC ROOKIE PATCH"
    image_url = "https://i.ebayimg.com/images/g/4w0AAOSwgTRmAuVJ/s-l1200.jpg"

    LLM_output = ebay_text_image_parse(ebay_text=ebay_text, image_url=image_url)
