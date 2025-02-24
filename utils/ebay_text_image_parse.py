import requests
import json
import pandas as pd
import re
from utils.call_predict_with_image import call_predict_with_image
from utils.program_cardSet_vecSearch import text_vecSearch
from utils.utils import preprocess_year_num, filter_dataframe_optimized, match_tag, sort_tags_by_text_position
from thefuzz import fuzz, process

Config_path = "Config.json"
Config = json.load(open(Config_path))

# LLM 服务器的地址 (根据你的实际情况修改)
LLM_API_URL = Config['LLM_API_URL']
VEC_SEARCH_PROGRAM_API_URL = Config['VEC_SEARCH_PROGRAM_API_URL']
VEC_SEARCH_CARD_SET_API_URL = Config['VEC_SEARCH_CARD_SET_API_URL']
VEC_SEARCH_ATHLETE_API_URL = Config['VEC_SEARCH_ATHLETE_API_URL']

program_cardSet_athlete_data = pd.read_csv(r"Data\program_cardSet_athlete.csv")
checklist_2023 = pd.read_csv(r"Data\checklist_2023.csv")
print('数据加载完成')


def judge_by_search_list(ebay_text: str, search_list: list[str], pass_word_list: list[str] = None):
    """
    # 用搜索的结果列表对比原文本
    """
    ebay_text = (ebay_text.replace('-', ' ')
                 .replace('.', ' ')
                 .replace('/', ' ')
                 .replace('’', ' ')
                 .lower())
    ebay_words = set(ebay_text.split())

    # 使用集合推导一次性转换并存储 pass_word_list
    pass_words = set(word.lower() for word in (pass_word_list or []))  # 处理 None 情况

    # 排序 (只在需要时排序)
    search_list.sort(key=lambda s: len(s.split()), reverse=True)
    tag_list = []

    for tag in search_list:
        temp_tag = tag
        tag = (tag.replace('.', ' ')
               .replace("'", ' ')
               .replace('’', ' ')
               .replace('-', ' ')
               .strip())
        tag_words = tag.lower().split()
        # all() 和生成器表达式，简洁高效, 检查生成器表达式中的所有条件是否都为 True。
        if all(
                word in ebay_words
                or (word.rstrip("s") in ebay_words)
                or (word + "s" in ebay_words)  # 处理 prizm 和 prizms 之类的字符
                for word in tag_words if word and word not in pass_words
        ):
            tag_list.append(temp_tag)
    if len(tag_list) != 0:
        return tag_list
    return False


def get_vec_search_judge_result(vec_search_url: str, ebay_text: str, vec_text: str,
                                pass_word_list: list[str] = None,
                                left_priority=False,
                                top_k=15):
    '''
    :param vec_search_url:
    :param ebay_text:
    :param vec_text:
    :param pass_word_list: 针对一些不影响匹配的单词, [base, and, - ]等
    :param left_priority: 针对program的, 如果有多个合适选项, 优先选最左边的
    :return:
    '''
    top_k_list = text_vecSearch(vec_search_url, vec_text, top_k=top_k)
    print(top_k_list)
    # program 严格判断
    tag_list = judge_by_search_list(ebay_text=ebay_text,
                                    search_list=top_k_list,
                                    pass_word_list=pass_word_list)
    # pass_word_list=['the', 'and'])
    # print('tag_list:', tag_list)
    if tag_list is not False:
        if left_priority:
            tag_list = sort_tags_by_text_position(ebay_text, tag_list)
            return tag_list[0]
        else:
            return tag_list[0]
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


def fuzz_search_by_checklist(checklist, ebay_text, compare_text, tag_name,
                             checklist_filter_data=None, pass_word_list: list = None):

    if checklist_filter_data is not None:
        filtered_data_list = list(filter_dataframe_optimized(checklist, checklist_filter_data)[tag_name])
    else:
        filtered_data_list = list(checklist[tag_name].dropna().unique())
    # 进行模糊搜索
    print('fuzz search text: [', compare_text, ']---|filter_len: ', len(filtered_data_list))
    matches_list = process.extract(query=compare_text,
                                   choices=filtered_data_list,
                                   scorer=fuzz.partial_ratio,
                                   limit=25)
    matches_list = [x[0] for x in matches_list]

    print('fuzz match: ', matches_list)
    judge_tag_list = judge_by_search_list(ebay_text=ebay_text,
                                          search_list=matches_list,
                                          pass_word_list=pass_word_list)

    if judge_tag_list is not False:
        return judge_tag_list[0]

    return ''


def ebay_text_image_parse_LLM(ebay_text, image_url):
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
                                                                 pass_word_list=program_pass_word_list,
                                                                 left_priority=True)

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
        vec_text = re.sub(re.escape(predict_athlete), '', vec_text, count=1, flags=re.IGNORECASE)

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
            vec_text = re.sub(re.escape(predict_program), '', vec_text, count=1, flags=re.IGNORECASE).strip()
            print('vec_text [去除program]: ', vec_text)

        predict_cardSet = get_vec_search_judge_result(vec_search_url=VEC_SEARCH_CARD_SET_API_URL,
                                                      ebay_text=ebay_text,
                                                      vec_text=vec_text,
                                                      pass_word_list=cardSet_pass_word_list)

    elif predict_program == '' and predict_cardSet != '':
        # vec_text = vec_text.replace(predict_cardSet, '').strip()
        vec_text = re.sub(re.escape(predict_cardSet), '', vec_text, count=1, flags=re.IGNORECASE).strip()
        print('vec_text [去除cardSet]: ', vec_text)
        predict_program = get_vec_search_judge_result(vec_search_url=VEC_SEARCH_PROGRAM_API_URL,
                                                      ebay_text=ebay_text,
                                                      vec_text=vec_text,
                                                      pass_word_list=program_pass_word_list)
    else:
        # vec_text = vec_text.replace(predict_program, '').strip()
        vec_text = re.sub(re.escape(predict_program), '', vec_text, count=1, flags=re.IGNORECASE).strip()
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


def ebay_text_image_parse(ebay_text):
    program_pass_word_list = ['the', 'and']
    cardSet_pass_word_list = ['base', 'and', 'set', '-']

    # 预处理去掉一个Panini
    ebay_text = re.sub(re.escape("Panini"), '', ebay_text, count=1, flags=re.IGNORECASE)
    # 预处理常见错字
    ebay_text = re.sub(re.escape("Mosiac"), 'Mosaic', ebay_text, flags=re.IGNORECASE)


    # 1 获取年份和编号
    preprocess_year_num_data = preprocess_year_num(ebay_text)
    print('1 获取年份和编号: ', preprocess_year_num_data['year'], preprocess_year_num_data['card_num'])

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

    # 2 LLM获取球员名称
    print('vec_text: ', vec_text)
    predict_athlete = get_vec_search_judge_result(vec_search_url=VEC_SEARCH_ATHLETE_API_URL,
                                                  ebay_text=ebay_text,
                                                  vec_text=vec_text,
                                                  top_k=15)
    if predict_athlete == '':
        predict_athlete = fuzz_search_by_checklist(checklist=checklist_2023,
                                                   ebay_text=vec_text,
                                                   compare_text=vec_text,
                                                   tag_name="athlete_new",
                                                   pass_word_list=cardSet_pass_word_list,
                                                   checklist_filter_data=None)

    if predict_athlete != '':
        vec_text = re.sub(re.escape(predict_athlete), '', vec_text, count=1, flags=re.IGNORECASE)
        print('获取球员名称: ', predict_athlete)

    # 3 向量搜索 program 和 card_set
    print('vec_text: ', vec_text)
    predict_program = get_vec_search_judge_result(vec_search_url=VEC_SEARCH_PROGRAM_API_URL,
                                                  ebay_text=ebay_text,
                                                  vec_text=vec_text,
                                                  pass_word_list=program_pass_word_list,
                                                  left_priority=True,
                                                  top_k=25)

    # 如果存在 program 那么从文本里去除 program
    if predict_program != '':
        vec_text = re.sub(re.escape(predict_program), '', vec_text, count=1, flags=re.IGNORECASE).strip()
        print('vec_text [去除program]: ', vec_text)

    # 4 根据checklist 重新筛选card_set
    checklist_input = {
        "program_new": predict_program,
        "card_num": preprocess_year_num_data['card_num'],
        "athlete_new": predict_athlete
    }
    # 当上面条件满足其2的适合才进行二次搜索
    predict_cardSet = ''
    research_flag = 0
    if predict_program != '':
        research_flag += 1
    if preprocess_year_num_data['card_num'].isdigit():
        research_flag += 1
    if predict_athlete != '':
        research_flag += 1

    if research_flag >= 2:
        predict_cardSet = fuzz_search_by_checklist(checklist=checklist_2023,
                                                   checklist_filter_data=checklist_input,
                                                   ebay_text=ebay_text,
                                                   compare_text=vec_text,
                                                   tag_name="card_set",
                                                   pass_word_list=cardSet_pass_word_list)
        print('predict_cardSet: ', predict_cardSet)

    if predict_cardSet == '':
        predict_cardSet = get_vec_search_judge_result(vec_search_url=VEC_SEARCH_CARD_SET_API_URL,
                                                      ebay_text=ebay_text,
                                                      vec_text=vec_text,
                                                      pass_word_list=cardSet_pass_word_list)
        print('模糊搜索失败, 向量搜索: ', predict_cardSet)

    output = {
        'year': preprocess_year_num_data['year'],
        'program': predict_program,
        'card_set': predict_cardSet,
        'card_num': preprocess_year_num_data['card_num'],
        'athlete': predict_athlete
    }

    print('++++结果++++: ', output)
    print('==' * 18)

    return output
