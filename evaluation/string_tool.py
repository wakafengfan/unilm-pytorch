import re

from configuration.config import *

blank_regexp = re.compile(r'\s+')
punctuation = set()


for line in (mrc_datset_path / "punctuation").open():
    punctuation.add(line.strip())


def drop_punctuation(string):
    """删除所有标点符号"""
    rstring = ""
    for char in string:
        if char not in punctuation:
            rstring += char
        else:
            rstring += " "
    return rstring


def split_string(string):
    split_tokens = []
    for char in string:
        split_tokens.append(char)
    return split_tokens


def strQ2B(string):
    """全角转半角"""
    rstring = ""
    for char in string:
        inside_code = ord(char)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring


def strB2Q(string, codec="utf8"):
    """半角转全角"""
    rstring = ""
    for char in string:
        inside_code = ord(char)
        if inside_code == 32:  # 半角空格直接转化
            inside_code = 12288
        elif 32 <= inside_code <= 126:  # 半角字符（除空格）根据关系转化
            inside_code += 65248

        rstring += chr(inside_code)
    return rstring.encode(codec, "ignore")


def filter_blank(string):
    return blank_regexp.sub('', string)


def filter_extra_blank(string):
    return blank_regexp.sub(' ', string)
