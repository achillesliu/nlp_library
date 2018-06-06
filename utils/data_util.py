from collections import Counter, defaultdict
import numpy as np


########################################################################################################################
# Converting functions.
########################################################################################################################

def remove_bie_in_string(tag):
    if len(tag.split('-')) == 1:
        return 'O'
    else:
        return tag[2:]


# noinspection PyPep8Naming
def return_in_tagset_or_O(tag, tagset):
    if tag in tagset:
        return tag
    else:
        return 'O'


def convert_tag_ner(tag, tagset):
    """
    Note that this only works in our special case (our way of annotating)
    1. In our ways of annotating, we have some srl tags mixed here.
    2. In some cases we also do not want to predict all the categories.

    The tagset serves as a prameter in the experiment.
    """
    if len(tag.split('-')) == 1:
        tag = 'O'
    else:
        ptag = tag.split('-', 1)[1]
        if ptag in tagset:
            tag = tag
        else:
            tag = 'O'
    return tag


def to_categorical_global(vec_str, str_index):
    """
    This together with the following is designed for multi-label-multi-clf.
    In this scenario we may miss some of the tags in converting to categorical array,
    if we use the naive function in the library.

    :param vec_str:
    :param str_index:
    :return:
    """
    arr = np.zeros([len(vec_str), len(str_index)])
    for indi, i in enumerate(vec_str):
        arr[indi][str_index[i]] = 1
    return arr


def to_categorical_global_parser(sent, parserindex):
    """
    This function is special for parser since each word's label is an list.
    """
    arr = np.zeros([len(sent), len(parserindex)])
    for indi, lst in enumerate(sent):
        for lbl in lst:
            arr[indi][parserindex[lbl]] = 1
    return arr


def combine_substructures(findss, flist, mode):
    """
    Given a flist (of string, eg words) and list of lists (findss, indicies of the former list),
    combine the flist into sub strings.

    :param findss:
    :param flist:
    :param mode: 'word', flist is a list of words, then take all of the sub list strings.
        'others', just take the first element of the sub list strings.
    :return:
    """
    flist = np.asarray(flist)
    flist2 = []
    ind = 0
    while ind < len(flist):

        # Check if the item is inside the findss.
        # If inside, add the combined elements to the new list.
        word_is_ner = False
        for inds in findss:
            if ind in inds:
                word_is_ner = True
                if mode == 'words':
                    flist2.append('_'.join(flist[inds[0]:inds[-1] + 1]))
                elif mode == 'others':
                    flist2.append(flist[inds[0]])
                ind += len(inds)
                break

        # If item is not in any of the findss, then just add this single to the new list.
        if word_is_ner:
            continue
        else:
            flist2.append(flist[ind])
            ind += 1
    return flist2

########################################################################################################################
# Converting functions end.
########################################################################################################################

########################################################################################################################
# Get Linguistic features.
########################################################################################################################


def get_linguistic_features(sents_of_wordlist, w2vmodel, spacymodel, posindex, parserindex, sents_of_ner=None, replace_word_with_ner=False):
    """
    return: list of linguistic features, one for each sentence.

    Padding is not included here.
    Since this linguistic features must integrate with other feature.
    Then padding should be done after the combination.

    If replace word with ner, in the word embedding lookup the word embedding
    is replaced with the word embedding of corresponding ner.
    """

    linguistic_m = []
    for sentwords, sentners in zip(sents_of_wordlist, sents_of_ner):
        sentvecs, sentposs, sentparsers = [], [], []

        sentstr = ' '.join(sentwords)
        sent = spacymodel(str(sentstr))

        for indw, word in enumerate(sent):

            # word embs
            if replace_word_with_ner:
                tmpner = sentners[indw]
                if tmpner != 'O':
                    changed_word = tmpner
                else:
                    changed_word = word.text
                sentvecs.append(w2vmodel[changed_word.lower()])
            else:
                sentvecs.append(w2vmodel[word.text.lower()])

            # pos tag
            sentposs.append(word.tag_)

            # parser
            if len(word.dep_.split('||')) >= 2:
                tmp = word.dep_.split('||')
            else:
                tmp = [word.dep_]
            sentparsers.append(tmp)

        sentvecs = np.asarray(sentvecs)
        onehotm_sentposs = to_categorical_global(sentposs, posindex)
        onehotm_sentparsers = to_categorical_global_parser(sentparsers, parserindex)

        linguistic_m.append(np.concatenate((sentvecs, onehotm_sentposs, onehotm_sentparsers), axis=-1))
    return linguistic_m


def check_sentner_legal(sentners, srl_ner):
    containlist = 0
    for ner in srl_ner:
        if isinstance(ner, list):  # special case that contain list, which is employs
            containlist = 'employs'
        elif ner == 'POPIB':  # special case that contain popib, which is transfer money
            containlist = 'transfer-money'

    if containlist == 0:
        return check_sentner_sub_legal(sentners, srl_ner)
    elif containlist == 'employs':
        srl_ner2 = [['Organization', 'Person'], ['Geopolitical_entity', 'Person']]
        legal = False
        for tmp in srl_ner2:
            if check_sentner_sub_legal(sentners, tmp):
                legal = True
                break
        return legal
    elif containlist == 'transfer-money':
        srl_ner2 = [[a, b] for a in ['Person', 'Organization', 'Product', 'Industry', 'BusinessUnit'] for b in
                    ['Person', 'Organization', 'Product', 'Industry', 'BusinessUnit']]
        legal = False
        for tmp in srl_ner2:
            if check_sentner_sub_legal(sentners, tmp):
                legal = True
                break
        return legal


def check_sentner_sub_legal(sentners, srl_ner):
    countersentners = Counter(sentners)
    hitcount = 0
    for ner in srl_ner:
        tofindners = [a + ner for a in ['S-', 'B-']]
        for key in countersentners:
            if countersentners[key] == 0:
                continue
            if key in tofindners:
                countersentners[key] -= 1
                hitcount += 1
                break
    if hitcount == len(srl_ner):
        return True
    else:
        return False