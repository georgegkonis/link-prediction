import re


class Description:
    text: str
    keywords: set[str]

    def __init__(self, text: str):
        self.text = text
        self.keywords = set(re.findall(r"''{0,5}(.*?)''{0,5}", text))

    def __str__(self):
        return f'text={self.text} \nkeywords={self.keywords}'

    def __repr__(self):
        return self.__str__()
