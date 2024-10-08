import json
import pandas as pd
import pyarrow as pa
import random
import os
import re

from tqdm import tqdm
from glob import glob
from collections import defaultdict, Counter

contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

manual_map = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]

def normalize_word(token):
    _token = token
    for p in punct:
        if (p + " " in token or " " + p in token) or (
            re.search(comma_strip, token) != None
        ):
            _token = _token.replace(p, "")
        else:
            _token = _token.replace(p, " ")
    token = period_strip.sub("", _token, re.UNICODE)

    _token = []
    temp = token.lower().split()
    for word in temp:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            _token.append(word)
    for i, word in enumerate(_token):
        if word in contractions:
            _token[i] = contractions[word]
    token = " ".join(_token)
    token = token.replace(",", "")
    return token

def merge_keys(d, old_key, existing_key):
    if old_key in d and existing_key in d:
        d[existing_key].extend(d.pop(old_key))

def make_arrow(root, dataset_root, single_plot=False, missing_type=None):
    GENRE_CLASS = ['Drama', 'Comedy', 'Romance', 'Thriller', 'Crime', 'Action', 'Adventure', 'Horror'
     , 'Documentary', 'Mystery', 'Sci-Fi', 'Fantasy', 'Family', 'Biography', 'War', 'History', 'Music',
     'Animation', 'Musical', 'Western', 'Sport', 'Short', 'Film-Noir']
    GENRE_CLASS_DICT = {}
    for idx, genre in enumerate(GENRE_CLASS):
        GENRE_CLASS_DICT[genre] = idx    

    image_root = os.path.join(root, 'images')
    label_root = os.path.join(root, 'labels')
    
    with open(f"{root}/split.json", "r") as fp:
        split_sets = json.load(fp)
        
    merge_keys(split_sets, 'dev', 'train')
    
    total_genres = []
    for split, samples in split_sets.items():
        data_list = []
        for sample in tqdm(samples):
            image_path = os.path.join(image_root, sample+'.jpeg')
            label_path = os.path.join(label_root, sample+'.json')
            with open(image_path, "rb") as fp:
                binary = fp.read()
            with open(label_path, "r") as fp:
                labels = json.load(fp)    
            
            # There could be more than one plot for a movie,
            # if single plot, only the first plots are used
            if single_plot:
                plots = [labels['plot'][0]]
            else:
                plots = labels['plot']
                
            genres = labels['genres']
            label = [1 if g in genres else 0 for g in GENRE_CLASS_DICT]
            data = (binary, plots, label, genres, sample, split)
            data_list.append(data)

        dataframe = pd.DataFrame(
            data_list,
            columns=[
                "image",
                "plots",
                "label",
                "genres",
                "image_id",
                "split",
            ],
        )
        
        # import pdb; pdb.set_trace()
        df_filtered = dataframe[dataframe['label'].apply(lambda x: sum([1 for i in x if i != 0]) == 1)]

        df_filtered['label'] = df_filtered['label'].apply(lambda x: x.index(1))
        df_filtered = df_filtered[df_filtered['label'] < 23]
        
        label_counts = df_filtered['label'].value_counts()
        top_8_labels = label_counts.nlargest(8).index
        
        label_mapping = {label: i for i, label in enumerate(top_8_labels)}

        df_top_8 = df_filtered[df_filtered['label'].isin(top_8_labels)]
        df_top_8['label'] = df_top_8['label'].map(label_mapping)
        
        
        table = pa.Table.from_pandas(df_top_8)
        
        
        # import pdb; pdb.set_trace()
        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/mmimdb_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)        

root = 'benchmark/RAW_DATA/IMDB'
dataset_saved = 'benchmark/RAW_DATA/IMDB/generate_arrows'
make_arrow(root, dataset_saved)