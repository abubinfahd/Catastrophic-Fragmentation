import re


def extract_pair(item):
    """Match Paper_Token.ipynb's dictionary traversal and Unicode selection."""
    values = []

    def visit(value):
        if isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, str):
            values.append(value)

    visit(item)
    en, bn = "", ""
    for value in values:
        value = value.strip()
        if not value:
            continue
        if not bn and re.search(r"[\u0980-\u09ff]", value):
            bn = value
        elif not en and re.search(r"[a-zA-Z]", value) and not re.search(r"[\u0980-\u09ff]", value):
            en = value
    return en, bn
