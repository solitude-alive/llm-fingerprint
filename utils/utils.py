import re


def match_and_extract(string):
    pattern = r"^([0-9\.]+)_([a-zA-Z]+)_([0-9a-zA-Z-]+)$"  # match the pattern "xxx_xxx_xxx" of the string
    match = re.match(pattern, string)
    if match:
        return match.groups()
    else:
        return None
