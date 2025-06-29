import re
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer

# Initialize Arabic NLP tools
arabic_stopwords = set(stopwords.words("arabic"))
stemmer = ISRIStemmer()

# Character normalization mapping
char_map = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا",
    "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي", "ـ": ""
})

# Compiled regex patterns
diacritics_pattern = re.compile(r"[\u064B-\u0652]")
punctuation_pattern = re.compile(r"[^\w\s]")
whitespace_pattern = re.compile(r"\s+")
repeated_char_pattern = re.compile(r"(.)\1+")
digits_pattern = re.compile(r'\d+')

def normalize_arabic(text):
    return text.translate(char_map)

def remove_diacritics(text):
    return diacritics_pattern.sub("", text)

def remove_punctuation(text):
    return punctuation_pattern.sub(" ", text)

def reduce_repeated_characters(text):
    return repeated_char_pattern.sub(r"\1", text)

def remove_stopwords(tokens):
    return [word for word in tokens if word not in arabic_stopwords]

def stem_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]

def preprocess_for_classification(text):
    if not isinstance(text, str):
        return ""

    text = text.strip().lower()
    text = normalize_arabic(text)
    text = remove_diacritics(text)
    text = remove_punctuation(text)
    text = reduce_repeated_characters(text)
    text = whitespace_pattern.sub(" ", text).strip()
    text = digits_pattern.sub('', text)

    tokens = text.split()
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)

    return " ".join(tokens)

def preprocess_for_summarization(text):
    if not isinstance(text, str):
        return ""

    text = text.strip().lower()
    text = remove_diacritics(text)
    text = whitespace_pattern.sub(" ", text).strip()
    text = digits_pattern.sub('', text)

    return text
