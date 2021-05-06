# -*- coding: utf-8 -*-
import re
import itertools
import unicodedata
from itertools import tee, islice, chain


DIGITS = {entry for entry in map(chr, range(48,58))} # '0-9'
ALPHABETS = {entry for entry in itertools.chain(map(chr, range(65,91)), map(chr, range(97,123)))} # 'A-Z,a-z'
NON_FORM_SYMBOLS = "⓪①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮?,"


def is_alpha_num_symbol(ch) :
    if ch in DIGITS : return True
    if ch in ALPHABETS : return True
    return False

def is_korean(ch) :
    '''
    check if korean characters
    see http://www.unicode.org/reports/tr44/#GC_Values_Table
    '''
    try:
        if 'HANGUL' in unicodedata.name(ch) : return True
    except (TypeError, ValueError):
        pass

    return False

def extract_formula_positions_from_text(text):
    positions = []
    formula_chars = []
    formula_idxs = []
    korean_chars = []

    for i, (prev_ch, curr_ch, next_ch) in enumerate(get_prev_and_next(text)):
        # (한글+빈칸)
        if is_korean(curr_ch) or (curr_ch is ' '):
            korean_chars.append(curr_ch)
        else:
            korean_chars = []

        formula_ = is_formula(prev_ch, curr_ch)
        if next_ch:
            next_formula_ = is_formula(curr_ch, next_ch)

        if formula_: # 수식이면
            if not((len(formula_chars) == 0) and (curr_ch is ' ')): # 빈칸 skip
                formula_chars.append(curr_ch)
                formula_idxs.append(i)

            # 다음이 수식이 아니거나, 수식의 끝이면
            if (not(next_formula_) and (len(formula_chars) >= 1)) or \
                    (((i+1) == len(text)) and (len(formula_chars) >= 1)):
                start_idx, end_idx = min(formula_idxs), max(formula_idxs)
                positions.append((start_idx, end_idx))

                formula_chars = []
                formula_idxs = []

    return positions

def get_prev_and_next(iterable):
    prev, curr, next = tee(iterable, 3)
    prev = chain([None], prev)
    next = chain(islice(next, 1, None), [None])
    return zip(prev, curr, next)

def is_formula(prev_ch, curr_ch):
    if prev_ch is None:
        return False

    # 수식 = 영문 || 기호(원문자X) || 빈칸
    formula_ = (curr_ch in ALPHABETS) or \
                    (not (is_korean(curr_ch)) and (curr_ch not in NON_FORM_SYMBOLS)) or \
                        (curr_ch is ' ')
    return formula_

def strip_text_by_positions(text, positions):
    rev_positions = sorted(positions, key=lambda x: x[0], reverse=True)

    for pos in rev_positions:
        start_pos, end_pos = (pos[0], pos[1]+1)
        extract_text = text[start_pos:end_pos]
        strip_text = extract_text.replace(' ', '')
        text = text.replace(extract_text, strip_text)

    return text

def replace_string_from_dict(str, replace_info):
    for key, val in replace_info.items():
        str = str.replace('{'+key.upper()+'}', val)
    str = str.replace('//', '/')
    return str


test_texts = [
    '⑤ \\frac{e}{2}-\frac{1}{2}',
    # 'f(x)=\\left\\{\\begin{array}{l}{ax}&{1<x<2)}\\\\{bx}&{(1<x<2)}\\end{array}\\right.',
    # '\\overset{\\frown}{상}\\\\대\\\\도\\\\\\underset{\\smile}{수}',
    # '(가)\\underline{\\qquad}',
    # '\\underbrace{1×1×1}_{49개}',
    # '② \\frac { 24 } { 5 } cm',
    # '다음 그림의 △ABC에서 \\overline { AB } ⊥ \\overline { CE }, \\overline { A C } ⊥ \\overline { B D }이고 ',
    # '\\overline { AB } = 20 cm, \\overline { AC } = 15 cm, \\overline { AD } : \\overline { DC } = 2 : 1일 때,',

]

if __name__ == '__main__':
    for text in test_texts:
        print("text : \n{}".format(text))
        positions = extract_formula_positions_from_text(text)
        for pos in positions:
            print(text[pos[0]:pos[1]+1])
        strip_text = strip_text_by_positions(text, positions)
        print(strip_text)