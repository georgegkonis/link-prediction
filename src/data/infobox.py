class Infobox:
    title: str | None
    content: dict[str, str]
    other: list[str]

    def __init__(self, text: str):
        self.title = None
        self.content = {}
        self.other = []

        for entry in text.split('|'):
            entry = entry.strip()
            key_value = entry.split('=', 1)

            if len(key_value) == 2:
                key, value = key_value
                if not key or not value or str.isspace(key) or str.isspace(value):
                    continue
                self.content[key.strip()] = value.strip()

            elif len(key_value) == 1:
                if not entry or str.isspace(entry):
                    continue
                elif entry.startswith('infobox'):
                    self.title = entry.replace('infobox', '').strip()
                else:
                    self.other.append(entry)

            else:
                raise ValueError(f'Error parsing key-value pair: {entry}')

    def __str__(self):
        return f'title={self.title} \ncontent={self.content} \nother={self.other}'

    def __repr__(self):
        return self.__str__()
