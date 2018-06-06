"""
This is cleaning the result after Wikiextractor.

"""

import bz2


def get_articles(filepath):

    lines = open(filepath).read().splitlines()
    articles = []
    article = []

    ind_line = 0
    while ind_line < len(lines):

        if not lines[ind_line]:
            article.append(' ')  # append a space to prevent nothing in two sentences
            ind_line += 1
            continue
        if lines[ind_line].startswith('<doc'):  # start of an article
            ind_line += 1  # skip this line since this is a mark not a sentence
            ind_line += 1  # skip this line since it's the title and not good for analysis
            continue
        elif lines[ind_line].startswith('</doc'):  # end of an article
            ind_line += 1  # same as above
            articles.append(''.join(article).strip())
            article = []
            continue

        article.append(lines[ind_line])
        ind_line += 1

    return articles


def clean_article(article_str, nlp):
    doc = nlp(article_str)
    doc_strs = []
    for sent in doc.sents:
        sent_words = []
        for token in sent:
            if token.is_punct:
                word = '#' + token.text
            else:
                word = token.text
            sent_words.append(word)
        sent_str = ' '.join(sent_words)
        doc_strs.append(sent_str)
    return '\n'.join(doc_strs)


# noinspection PyUnboundLocalVariable
def read_one_page(fpath):
    if fpath.endswith('.bz2'):
        lines = bz2.open(fpath).read().decode().splitlines()
    return lines
