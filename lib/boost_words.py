import json
import os


def load_boost_words(boost_file, boost_args):
    boost_words = {}
    if boost_file and os.path.exists(boost_file):
        with open(boost_file) as f:
            boost_words = json.load(f)
    for entry in boost_args:
        if ":" in entry:
            word, factor = entry.rsplit(":", 1)
            boost_words[word] = float(factor)
        else:
            boost_words[entry] = 1.5
    return boost_words
