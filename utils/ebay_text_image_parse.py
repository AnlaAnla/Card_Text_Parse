import requests
import json
import pandas as pd
import re
from utils.call_predict_with_image import call_predict_with_image
from utils.program_cardSet_vecSearch import text_vecSearch

Config_path = "Config.json"
Config = json.load(open(Config_path))

# LLM 服务器的地址 (根据你的实际情况修改)
LLM_API_URL = Config['LLM_API_URL']
VEC_SEARCH_PROGRAM_API_URL = Config['VEC_SEARCH_PROGRAM_API_URL']
VEC_SEARCH_CARD_SET_API_URL = Config['VEC_SEARCH_CARD_SET_API_URL']

program_cardSet_athlete_data = pd.read_csv("Data\program_cardSet_athlete.csv")


def preprocess_year_num(text):
    # 尝试匹配 "20xx-yy"、"20xx" 或 "xx-yy" 格式的年份
    year_match = re.search(r'\b(20\d{2})-(\d{2})\b|\b(20\d{2})\b|\b(\d{2})-(\d{2})\b', text)
    num_match = re.search(r'#([A-Za-z0-9\-]+)', text)

    if year_match:
        # "20xx-yy" 格式 (group 1 和 2)
        if year_match.group(1):
            year = year_match.group(1)
        # "20xx" 格式 (group 3)
        elif year_match.group(3):
            year = year_match.group(3)
        # "xx-yy" 格式 (group 4 和 5)
        elif year_match.group(4):
            year = "20" + year_match.group(4)  # 添加 "20" 前缀
            # 简单检查，防止 "99-00" 这样的情况被错误处理
            if int(year) > 2099 or int(year) < 2000:
                year = ""

        else:
            year = ""
    else:
        year = ""

    return {
        'year': year,
        'card_num': num_match.group(1) if num_match else ""
    }


def match_tag(csv_data, tag, text):
    # 和数据库匹配
    matches = csv_data[csv_data[tag].str.contains(f"^{text}$", na=False, regex=True, case=False)]
    return not matches.empty


def judge_by_vec_search_list(ebay_text: str, vector_list: list[str], pass_word_list: list[str] = None):
    """
    # 用向量搜索的结果对比原文本
    """
    ebay_text = ebay_text.replace('-', ' ').replace('.', ' ').replace('/', ' ').lower()
    ebay_words = set(ebay_text.split())

    # 使用集合推导一次性转换并存储 pass_word_list
    pass_words = set(word.lower() for word in (pass_word_list or []))  # 处理 None 情况

    # 排序 (只在需要时排序)
    vector_list.sort(key=lambda s: len(s.split()), reverse=True)

    for tag in vector_list:
        tag_words = tag.lower().split()
        # all() 和生成器表达式，简洁高效, 检查生成器表达式中的所有条件是否都为 True。
        if all(
            word in ebay_words
            or (word.rstrip("s") in ebay_words)
            or (word + "s" in ebay_words)  # 处理 prizm 和 prizms 之类的字符
            for word in tag_words if word and word not in pass_words
        ):
            return tag
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


def judge_tag_in_text(ebay_text: str, tag: str, pass_word_list: list = None):
    ebay_text = ebay_text.replace('-', ' ').replace('.', ' ').replace('/', ' ').lower()
    ebay_words = set(ebay_text.split())

    # 使用集合推导一次性转换并存储 pass_word_list
    pass_words = set(word.lower() for word in (pass_word_list or []))  # 处理 None 情况
    tag_words = tag.lower().split()
    if all(
        word in ebay_words
        or (word.rstrip("s") in ebay_words)
        or (word + "s" in ebay_words)  # 处理 prizm 和 prizms 之类的字符
        for word in tag_words if word and word not in pass_words
    ):
        return True
    return False

    # for i in range(len(pass_word_list)):
    #     # 全部小写
    #     pass_word_list[i] = pass_word_list[i].lower()

    # is_include_flag = False
    # for word in tag.split(' '):
    #     if word.lower() in pass_word_list or word == '':
    #         continue
    #
    #     if word.lower() in ebay_text.lower().split(' '):
    #         is_include_flag = True
    #     else:
    #         is_include_flag = False
    #         break
    # return is_include_flag


def ebay_text_image_parse(ebay_text, image_url):
    program_pass_word_list = ['the', 'and']
    cardSet_pass_word_list = ['base', 'and', 'set', '-']

    # 预处理去掉一个Panini
    ebay_text = re.sub(re.escape("Panini"), '', ebay_text, count=1, flags=re.IGNORECASE)

    # 1 获取年份和编号
    preprocess_year_num_data = preprocess_year_num(ebay_text)
    print('1 获取年份和编号: ', preprocess_year_num_data['year'], preprocess_year_num_data['card_num'])

    # 2 LLM获取球员名称
    LLM_output = call_predict_with_image(LLM_API_URL, image_url, ebay_text)

    print()
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
        program_is_match = (match_tag(program_cardSet_athlete_data, tag='program', text=predict_program) and
                            judge_tag_in_text(ebay_text=ebay_text,
                                              tag=predict_program,
                                              pass_word_list=program_pass_word_list))

        # 精确匹配后, 用向量匹配重新匹配, 取正确中最长的作为结果
        predict_program_vec_result = get_vec_search_judge_result(vec_search_url=VEC_SEARCH_PROGRAM_API_URL,
                                                                 ebay_text=ebay_text,
                                                                 vec_text=predict_program,
                                                                 pass_word_list=program_pass_word_list)

        if program_is_match and predict_program_vec_result != '':
            predict_program = predict_program_vec_result \
                if len(predict_program_vec_result) > len(predict_program) else predict_program
        elif program_is_match and predict_program_vec_result == '':
            predict_program = predict_program
        elif not program_is_match and predict_program_vec_result != '':
            predict_program = predict_program_vec_result
        else:
            predict_program = ''

        if predict_program != '':
            print('3 获取 programe: ', predict_program)

    if predict_cardSet != '':
        cardSet_is_match = (match_tag(program_cardSet_athlete_data, tag='card_set', text=predict_cardSet) and
                            judge_tag_in_text(ebay_text=ebay_text,
                                              tag=predict_cardSet,
                                              pass_word_list=cardSet_pass_word_list))

        # 精确匹配后, 用向量匹配重新匹配 ,取正确中最长的作为结果
        predict_cardSet_vec_result = get_vec_search_judge_result(vec_search_url=VEC_SEARCH_CARD_SET_API_URL,
                                                                 ebay_text=ebay_text,
                                                                 vec_text=predict_cardSet,
                                                                 pass_word_list=cardSet_pass_word_list)

        if cardSet_is_match and predict_cardSet_vec_result != '':
            predict_cardSet = predict_cardSet_vec_result \
                if len(predict_cardSet_vec_result) > len(predict_cardSet) else predict_cardSet
        elif cardSet_is_match and predict_cardSet_vec_result == '':
            predict_cardSet = predict_cardSet
        elif not cardSet_is_match and predict_cardSet_vec_result != '':
            predict_cardSet = predict_cardSet_vec_result
        else:
            predict_cardSet = ''

        if predict_cardSet != '':
            print("3 获取 card_set: ", predict_cardSet)

    # 二次验证card_set, 临时存储数据
    temp_save_cardSet = predict_cardSet
    predict_cardSet = ''

    if predict_program != '' and predict_cardSet != '':
        LLM_output['program'] = predict_program
        LLM_output['card_set'] = predict_cardSet

        print('_' * 20)
        print(ebay_text)
        print("LLM : ", LLM_output)
        print('++++结果++++: ', LLM_output)
        return LLM_output

    # 4 向量搜索 program , card_set
    print('_' * 20)
    vec_text = ebay_text
    if preprocess_year_num_data['year'] != '':
        # 消除年份, 这样可以消除 2023-24 这种格式
        vec_text = (vec_text
                    .replace(re.search(r'\b(20\d{2})-(\d{2})\b|\b(20\d{2})\b|\b(\d{2})-(\d{2})\b', vec_text)
                             .group(), '').strip())
    if preprocess_year_num_data['card_num'] != '':
        vec_text = vec_text.replace(preprocess_year_num_data['card_num'], '')
        vec_text = vec_text.replace('#', '')
    if predict_athlete != '':
        # vec_text = vec_text.replace(predict_athlete, '')
        vec_text = re.sub(re.escape(predict_athlete), '', vec_text, count=1,flags=re.IGNORECASE)

    # 在这里根据 program 和 card_set 的有无分为三种情况
    if predict_program == '' and predict_cardSet == '':
        print('vec_text: ', vec_text)
        predict_program = get_vec_search_judge_result(vec_search_url=VEC_SEARCH_PROGRAM_API_URL,
                                                      ebay_text=ebay_text,
                                                      vec_text=vec_text,
                                                      pass_word_list=program_pass_word_list)

        # 如果存在 program 那么从文本里去除 program
        if predict_program != '':
            # vec_text = vec_text.replace(predict_program, '').strip()
            vec_text = re.sub(re.escape(predict_program), '', vec_text, count=1,flags=re.IGNORECASE).strip()
            print('vec_text [去除program]: ', vec_text)

        predict_cardSet = get_vec_search_judge_result(vec_search_url=VEC_SEARCH_CARD_SET_API_URL,
                                                      ebay_text=ebay_text,
                                                      vec_text=vec_text,
                                                      pass_word_list=cardSet_pass_word_list)

    elif predict_program == '' and predict_cardSet != '':
        # vec_text = vec_text.replace(predict_cardSet, '').strip()
        vec_text = re.sub(re.escape(predict_cardSet), '', vec_text, count=1,flags=re.IGNORECASE).strip()
        print('vec_text [去除cardSet]: ', vec_text)
        predict_program = get_vec_search_judge_result(vec_search_url=VEC_SEARCH_PROGRAM_API_URL,
                                                      ebay_text=ebay_text,
                                                      vec_text=vec_text,
                                                      pass_word_list=program_pass_word_list)
    else:
        # vec_text = vec_text.replace(predict_program, '').strip()
        vec_text = re.sub(re.escape(predict_program), '', vec_text, count=1,flags=re.IGNORECASE).strip()
        print('vec_text [去除program]: ', vec_text)
        predict_cardSet = get_vec_search_judge_result(vec_search_url=VEC_SEARCH_CARD_SET_API_URL,
                                                      ebay_text=ebay_text,
                                                      vec_text=vec_text,
                                                      pass_word_list=cardSet_pass_word_list)

    print('_' * 20)
    print(ebay_text)
    print("LLM : ", LLM_output)
    # print('LLM2:', LLM_output2)

    LLM_output['program'] = predict_program

    if len(predict_cardSet) > len(temp_save_cardSet):
        LLM_output['card_set'] = predict_cardSet
    else:
        LLM_output['card_set'] = temp_save_cardSet

    print('++++结果++++: ', LLM_output)
    print('==' * 18)

    return LLM_output
