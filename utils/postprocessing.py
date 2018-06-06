

def postprocess_ner(sentwords, nersinsent):
    """
    Convert the predicted list of ners ([B-Person, E-Person, O, O, I-Product]) into the word-ner dict.
    Specially designed for BIESO annotation.

    :param sentwords:
    :param nersinsent:
    :return:
    """
    ners, words, indofner = [], [], []
    candiners, candiwords, tmpinds = [], [], []
    for indn, num in list(enumerate(nersinsent)):

        # Start a new pool. The pool is only for B. For S the word and ner would be immediately append and no pool created.
        if (not candiners) and (not candiwords):
            currentner = nersinsent[indn]
            if currentner == 'O':
                continue
            elif currentner[:2] in ['I-', 'E-']:
                continue
            elif currentner[:2] == 'S-':
                ners.append(currentner[2:])
                words.append(sentwords[indn])
                indofner.append([indn])
                continue
            else:
                tmpinds.append(indn)
                candiners.append(currentner)
                candiwords.append(sentwords[indn])

        # Link the next word with the existing pool.
        if candiners and candiwords and (indn + 1 < len(sentwords)):
            currentner = nersinsent[indn]
            nextner = nersinsent[indn+1]
            if nextner == 'O':  # wrong sequence, o followed b.
                candiners, candiwords = [], []
                continue
            if nextner[2:] != currentner[2:]:  # wrong ners, the nertags are different.
                candiners, candiwords = [], []
                continue
            else:
                if nextner[:2] in ['B-', 'S-']:  # wrong sequence, b or s follows b.
                    candiners, candiwords = [], []
                    continue
                elif nextner[:2] == 'E-':  # e followed b.
                    candiners.append(nextner)
                    candiwords.append(sentwords[indn + 1])
                    tmpinds.append(indn+1)
                    ners.append(currentner[2:])
                    words.append(' '.join(candiwords))
                    indofner.append(tmpinds)
                    candiners, candiwords, tmpinds = [], [], []
                else:
                    tmpinds.append(indn+1)
                    candiners.append(nextner)
                    candiwords.append(sentwords[indn + 1])
    return ners, words, indofner
