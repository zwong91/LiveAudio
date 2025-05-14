import base64

def image_bytes_to_base64_data_uri(img_bytes: bytes, format: str = "jpeg", encoding="utf-8"):
    base64_data = base64.b64encode(img_bytes).decode(encoding)
    return f"data:image/{format};base64,{base64_data}"


def image_to_base64_data_uri(file_path: str, format: str = "jpeg", encoding="utf-8"):
    with open(file_path, "rb") as img_file:
        return image_bytes_to_base64_data_uri(img_file.read(), format=format, encoding=encoding)


import re
from typing import List

# 常见英文缩写（小写）——可按需补充
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
    "e.g", "i.e", "etc", "vs", "st", "ft", "inc", "ltd"
}

# 正则：匹配所有可能作为终结符的标点
_TERMINATORS = r"""
    (?P<dot>\.{1,3})       |   # 英文句点 1~3 个（包括英文省略号 ...）
    (?P<zh_ellipsis>…{1,2})|   # 中文省略号 … 或 …
    (?P<end>[。！？!?])         # 中英文问号、叹号、句号
"""

_TERMINATOR_RE = re.compile(_TERMINATORS, re.UNICODE | re.VERBOSE)

def smart_split(text: str) -> List[str]:
    """
    智能分句：按中英文终结符（。！？…!?、英文省略号... 等）断句，
    跳过常见英文缩写和数字小数点，不在它们内部切分。
    """
    sentences = []
    start = 0

    for m in _TERMINATOR_RE.finditer(text):
        end = m.end()
        segment = text[start:end]

        # 如果是单个点，需要检查是否是缩写或小数点
        if m.group("dot") == ".":
            # 看前面连续字符（单词或数字）
            token = re.search(r'([A-Za-z]+|\d+)\.$', segment)
            if token:
                t = token.group(1)
                # 缩写过滤
                if t.lower() in _ABBREVIATIONS:
                    continue
                # 数字小数点过滤（如 3.14 的 .14）
                if t.isdigit():
                    continue

        sentences.append(segment.strip())
        start = end

    # 剩余部分
    if start < len(text):
        tail = text[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences
