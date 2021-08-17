import os
import re


def regex_glob(directory: str, file_regex: str):
    return [
        f"{directory}/{f}" for f in os.listdir(directory) if re.search(file_regex, f)
    ]
