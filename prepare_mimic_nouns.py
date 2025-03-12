import re 
from textblob import TextBlob, Word
from tqdm import tqdm

_DISCARD_WORDS = ['photo', 'background', 'stock', 'image', 'closeup', 'jpg', 'picture', 'png', 'file', 'close up', 'pictures', 'ive', 'view', 'www', 'http', 'showing', 'blurred', 'shot', 'example', 'camera', 'footage', 'free', 'video', 'displaying', 'display', 'displayed', 'thumbnail', 'focus', 'focusing', 'detail', 'panoramic', 'standard', 'portrait', 'zoom', 'zoomed', 'show', 'showed', 'real', 'icon', 'pixelated', 'cropped', 'autofocus', 'caption', 'graphic', 'defocused', 'zoomed', ' pre ', 'available', 'royalty', 'etext', 'blurry', 'new', 'pic', 'left', 'houzz', 'full', 'small', 'br', 'looking', 'pro', 'angle', 'logo', 'close', 'right', 'blur', 'preview', 'wallpaper', 'dont', 'fixed', 'closed', 'open', 'profile', 'close', 'color', 'photo', 'colored', 'video', 'banner', 'macro', 'frame', 'cut', 'livescience', 'bottom', 'corner', 'tvmdl', 'overlay', 'original', 'sign', 'old', 'extreme', 'hq', 'isolated', 'figure', 'stockfoto', 'vrr', 'cm', 'photography', 'print', 'embedded', 'smaller', 'testing', 'captioned', 'year', 'photograph', '', 'selective', 'photoshopped', 'come', 'org', 'akc', 'iphone']
def clean_text(text):
    text = text.lower()
    text = re.sub("'", '', text)
    text = re.sub("[^A-Za-z0-9 \n']+", ' ', text)
    text = re.sub('fig\d+', ' ', text)
    text = re.sub(' . ', ' ', text)
    text = ' '.join([t for t in text.split(' ') if t not in _DISCARD_WORDS])
    return text

report_path = './dataset/report/mimic_test_unique.txt'
nouns_path = f'./dataset/report/nouns.txt'

with open(report_path, 'r') as f: 
    report = (f.read()).split('\n')

all_noun_phrases = []
all_words = []
all_nouns = []
all_verbs = []
all_adj = []

for cp in tqdm(range(len(report))):
    cap = report[cp]
    cap = clean_text(cap)
    blob = TextBlob(cap)

    noun_phrases_in_cap = blob.noun_phrases
    words_in_cap = set([i[0] for i in blob.tags if i[1] in ['NN', 'NNP', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS']])
    nouns_in_cap = set([i[0] for i in blob.tags if i[1] in ['NN', 'NNP', 'NNS']])
    verbs_in_cap = set([i[0] for i in blob.tags if i[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']])
    adj_in_cap = set([i[0] for i in blob.tags if i[1] in ['JJ', 'JJR', 'JJS']])
    # words_in_cap = blob.words

    noun_phrases_temp = [k for k in list(noun_phrases_in_cap) if k not in all_noun_phrases]
    all_noun_phrases = all_noun_phrases + noun_phrases_temp

    words_in_cap_temp = [k for k in list(words_in_cap) if k not in all_words]
    all_words = all_words + words_in_cap_temp

    nouns_in_cap_temp = [k for k in list(nouns_in_cap) if k not in all_nouns]
    all_nouns = all_nouns + nouns_in_cap_temp

    verbs_in_cap_temp = [k for k in list(verbs_in_cap) if k not in all_verbs]
    all_verbs = all_verbs + verbs_in_cap_temp

    adj_in_cap_temp = [k for k in list(adj_in_cap) if k not in all_adj]
    all_adj = all_adj + adj_in_cap_temp

with open(nouns_path, 'w+') as f:
    f.write('\n'.join(all_nouns))