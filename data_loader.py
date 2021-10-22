import torch
import glob
import os
import string
import unicodedata


all_letters = string.ascii_letters + " .,;'"
category_lines = {}
all_categories = []

def files_list(path):
    return glob.glob(path)


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, len(all_letters))
    for n, char in enumerate(line):
        tensor[n][0][all_letters.find(char)] = 1
    return tensor


for filename in files_list('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = read_lines(filename)
    category_lines[category] = lines
