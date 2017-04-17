import nltk
import random
from nltk.stem import *
from os.path import join
from math import floor
import numpy as np
import Data


class Normalize:

    def __init__(self):
        self._accept_chars = {}
        for c in [chr(i) for i in range(ord('0'), ord('9')+1)]:
            self._accept_chars[c] = c
        for c in [chr(i) for i in range(ord('a'), ord('z')+1)]:
            self._accept_chars[c] = c

        self._apost = {
            '\'s': '',
            '\'d': 'would',
            '\'ll': 'will',
            '\'ve': 'have',
            '\'re': 'are',
            'n\'t': 'not',
            '\'cause': 'because'
        }

        # 0 -> _ + e
        # 1 -> _::tail -> _ + y
        # 2 -> _ + y
        # 3 -> _ + ing
        # 4 -> _ + al
        # 5 -> _ + s
        # 6 -> _ + er
        # 7 -> _ + ate
        # 8 -> _::const::tail -> _ + const + const + ing
        # 9 -> _ + ed
        self._rect = {
            'absenc': 0,
            'abus': 0,
            'accus': 0,
            'awar': 0,
            'amaz': 0,
            'announc': 0,
            'anymor': 0,
            'becom': 0,
            'behavior': 0,
            'believ': 0,
            'continu': 0,
            'caus': 0,
            'debat': 0,
            'describ': 0,
            'despit': 0,
            'determin': 0,
            'diagnos': 0,
            'els': 0,
            'execut': 0,
            'headlin': 0,
            'knowledg': 0,
            'misl': 0,
            'notic': 0,
            'offic': 0,
            'outrag': 0,
            'peopl': 0,
            'relat': 0,
            'revenu': 0,
            'scienc': 0,
            'sinc': 0,
            'sourc': 0,
            'somewher': 0,
            'spokespeopl': 0,
            'vacat': 0,
            'valu': 0,
            'youtub': 0,

            'centuri': 1,
            'compani':1,
            'everi': 1,
            'famili': 1,
            'frenzi': 1,
            'histori': 1,
            'memori': 1,
            'middl': 1,
            'onli': 1,
            'pri': 1,
            'readi': 1,
            'shini': 1,
            'studi': 1,
            'tri': 1,

            'disagr': 2,
            'memor': 2,

            'anyth': 3,

            'controversi': 4,
            'financi': 4,
            'individu': 4,

            'goe': 5,

            'anoth': 6,
            'howev': 6,

            'inappropri': 7,
            'anticip': 7,

            'wedi': 8,

            'inde': 9
        }

        self._stemmer = SnowballStemmer('english')

    def calonialize(self, w):
        if w in self._apost:
            return self._apost[w]
        return w

    def remove_punct(self, w):
        buffer = ''
        for c in w:
            if c in self._accept_chars:
                buffer += c
        return buffer

    def stem(self, w):
        return self._stemmer.stem(w)

    def rectify(self, w):
        """ hard code cases where stem is wrong"""
        if w in self._rect:
            code = self._rect[w]
            if code == 0:
                return w + 'e'
            elif code == 1:
                return w[:-1] + 'y'
            elif code == 2:
                return w + 'y'
            elif code == 3:
                return w + 'ing'
            elif code == 4:
                return w + 'al'
            elif code == 5:
                return w + 's'
            elif code == 6:
                return w + 'er'
            elif code == 7:
                return w + 'ate'
            elif code == 8:
                return w[:-1] + w[-2] + 'ing'
        return w

    def norm(self, s):
        s = nltk.word_tokenize(s.lower())
        s = [self.calonialize(w) for w in s]
        s = [self.remove_punct(w) for w in s]
        s = [self.stem(w) for w in s if w != '']
        s = [self.rectify(w) for w in s]
        return s


class Node(object):
    Word = ''
    Theta = {}
    Index = -1
    Edges = []

    def __init__(self, w='', t=None, i=-1, e=None):
        self.Theta = t if t is not None else {}
        self.Word = w
        self.Index = i
        self.Edges = e if e is not None else []

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return other.Index == self.Index

    def compare(self, other) -> bool:
        res = 0
        for sub in self.Theta:
            if sub in other.Theta:
                res += self.Theta[sub]
            else:
                res -= self.Theta[sub]
        for sub in other.Theta:
            if sub in self.Theta:
                res += other.Theta[sub]
            else:
                res -= other.Theta[sub]
        return res


class SuperNode(Node):

    def add(self, node: Node) -> None:
        if node in self.Edges:
            err = 'WTF bro no repeats!\n'
            err += 'Attempting to add duplication of "{0}" into list:\n'
            err += str([w for w in [e.Word for e in self.Edges]])
            raise Exception(err)

        # build a running total of the values
        for sub in node.Theta.keys():
            if sub not in self.Theta:
                self.Theta[sub] = 0
            self.Theta[sub] *= len(self.Edges)
            self.Theta[sub] += node.Theta[sub]
            self.Theta[sub] /= len(self.Edges) + 1

        self.Edges.append(node)
        self._gen_best_fit()

    def _hit_score(self, other):
        score = 0
        for sub in other.Theta:
            score = 1 if sub in self.Theta else -1
        return score

    def _gen_best_fit(self) -> None:
        best_score = 0
        best_word = ''

        for curr in self.Edges:
            score = self._hit_score(curr)
            if score > best_score:
                best_score = score
                best_word = curr.Word

        self.Word = best_word


class Bucket:

    def __init__(self):
        self._suffix = [
            ['s', 'y', 'i'],
            ['es', 'ed', 'ly', 'er', 'or', 'al', 'ty', 'ie', 'en', 'ar'],
            ['ing', 'ion', 'ial', 'ity', 'ous', 'ive', 'ful', 'est', 'ish', 'ess'],
            ['tion', 'sion', 'ible', 'able', 'ness', 'ment', 'eous', 'ious', 'less', 'ette', 'like', 'ship'],
            ['ation', 'ition', 'ative', 'itive']
        ]
        self._weights = [0.3, 1., 4., 8., 12.]

        self._supernodes = []
        self._nodes = []
        words = [w.strip('\n') for w in open(join('words', 'wordsEn.txt')).readlines() if w != '']

        for i in range(len(words)):
            self._nodes.append(Node(words[i], self.theta(words[i]), i))
        self._num_nodes = len(self._nodes)
        del words

    def theta(self, word: str) -> {str, int}:
        subs = {}
        for r in range(4):
            w_len = len(word)
            if w_len < r:
                return subs
            weight = self._weights[r - 1]

            for i in range(w_len - r + 1):
                sub = word[i:i + r]
                if sub == '':
                    continue
                if sub not in subs:
                    subs[sub] = 0
                if i == w_len - r:
                    if sub in self._suffix[r - 1] and w_len > r + 2:
                        subs[sub] += weight / r
                    else:
                        subs[sub] += weight
                else:
                    subs[sub] += weight

        return subs

    def findClosest(self, index, rng) -> [(float, int)]:
        # basing shit on the fact that the words are assorted.
        # we dont need to look at the whole list... thank god
        _min = index - 10 if index - 10 > 0 else 0
        _max = index + 10 if index + 10 < self._num_nodes else self._num_nodes - 1

        xs = []
        curr = self._nodes[index]
        for idx in range(_max - _min):
            if _min + idx == index: continue

            other = self._nodes[_min + idx]
            score = curr.compare(other)
            xs.append((score, _min + idx))
            xs = sorted(xs, key=lambda tup: (-tup[0], tup[1]))[:rng]

        return xs

    def bin_find(self, node) -> int:
        lhs, rhs = 0, len(self._supernodes) - 1
        if rhs - lhs < 0: return -1

        while True:
            peek = int(lhs + ((rhs - lhs) / 2))
            if rhs == lhs: return rhs

            _super = self._supernodes[peek]
            if _super.compare(node) > 0:
                return peek
            elif _super.Word > node.Word:
                rhs = peek
            else:
                lhs = peek

    def step(self) -> [SuperNode]:
        res = []
        for i in range(self._num_nodes):
            res.append((i, self.findClosest(i, 15)))

        for _i, _set in res:
            _curr = self._nodes[_i]
            sn = SuperNode()
            if len(_curr.Edges) == 0:
                sn.add(_curr)

            for _score, _oi in _set:
                _other = self._nodes[_oi]
                if (_score > 0) and (_i not in _other.Edges):
                    sn.add(_other)
                    _curr.Edges.append(_oi)
                    _other.Edges.append(_i)

            if len(sn.Edges) > 0:
                self._supernodes.append(sn)

        return self._supernodes


class Bucket2(object):

    def __init__(self):

        self._sub_sizes = (2, 3)

        self._weights = {
            1: Data.weights_sub_1(),
            2: Data.weights_sub_2(),
            3: Data.weights_sub_3()
        }

        self._bias = {
            1: 1.100000381469727,
            2: 3.510741233825684,
            3: 6.238486766815186,
            4: 5.255193710327148,
            5: 1.000000476837158
        }

        #self._bias = {
        #    1: 1,
        #    2: 1.5,
        #    3: 2,
        #}

        # first elem in list is the Average, rest is set
        # [(str, {str, float})]
        self._family = []

    def _theta(self, word: str) -> {str, float}:
        subs = {}
        for r in self._sub_sizes:
            w_len = len(word)
            if w_len < r:
                return subs
            weight = self._weights[r]
            bias = self._bias[r]

            for i in range(int(w_len - r + 1)):
                sub = word[i:i + r]
                if sub == '':
                    continue
                if sub not in subs:
                    subs[sub] = 0

                if sub in weight:
                    subs[sub] += (weight[sub] * bias)
                else:
                    subs[sub] += bias
        return subs

    def _theta_prime(self, word: str) -> {str, float}:
        subs = {}
        for r in self._sub_sizes:
            w_len = len(word)
            if w_len < r:
                return subs

            for i in range(int(w_len - r + 1)):
                sub = word[i:i + r]
                if sub == '':
                    continue
                if sub not in subs:
                    subs[sub] = 0
                subs[sub] += 1
        return subs

    @staticmethod
    def _compare(w1: (str, {str, float}),
                 w2: (str, {str, float})) -> float:
        res = 0
        w1, d1 = w1
        w2, d2 = w2
        for sub in d1:
            res += d1[sub] if sub in d2 else -d1[sub]
        for sub in d2:
            res += d2[sub] if sub in d1 else -d2[sub]
        return res

    @staticmethod
    def _hit_score(w1: (str, {str, float}), w2: (str, {str, float})) -> int:
        score = 0
        for sub in w2[1]:
            score = 1 if sub in w1[1] else -1
        return score

    @staticmethod
    def _gen_base_word(words: [(str, {str, float})]) -> str:
        best_score = -99999
        best_word = ''

        for word in words:
            if best_score < -len(word[0]):
                best_score = -len(word[0])
                best_word = word[0]

        return best_word

    def _bin_find(self, word: (str, {str, float})) -> (bool, int):
        lhs, rhs = 0, len(self._family) - 1
        if rhs - lhs < 0:
            return False, -1

        while True:
            peek = int(lhs + floor((rhs - lhs) / 2))

            fam_word = self._family[peek][0]
            if self._compare(fam_word, word) > 0:
                return True, peek

            elif rhs == lhs:
                if fam_word[0] < word[0]:
                    peek += 1
                    if peek == len(self._family):
                        return False, -1
                return False, peek

            elif fam_word[0] > word[0]:
                rhs = peek
            else:
                lhs = peek + 1

    def _add_to_fam(self, index: int, word: (str, {str, float})) -> bool:
        if index == -1:
            self._family.append([word, word])
            return True

        if index < 0 or index > len(self._family):
            return False

        fam = self._family[index]
        av_word, av_theta = fam[0]
        n = len(av_theta)

        # running average of substring counts
        for _sub in av_theta:
            av_theta[_sub] *= n
            if _sub in word[1]:
                av_theta[_sub] += word[1][_sub]
            av_theta[_sub] /= (n + 1)

        # generate base word
        fam.append(word)
        av_word = Bucket2._gen_base_word(fam[1:])
        fam[0] = (av_word, av_theta)

        del self._family[index]
        found, index = self._bin_find(fam[0])
        if index < 0:
            self._family.append(fam)
        else:
            self._family = self._family[:index] + [fam] + self._family[index:]

        return True

    def _insert_to_fam(self, index: int, word: (str, {str, float})) -> bool:
        if index < 0 or index > len(self._family):
            return False

        self._family = self._family[:index] + [[word, word]] + self._family[index:]
        return True

    def add_word(self, word: str) -> bool:
        """
        Look for a word family in database, otherwise create
        new entry. Return the base word for that family.
        
        :param text: str -> text to be evaluated 
        :return: str -> the base version of that word
        """
        wordt = [word, self._theta(word)]
        found, index = self._bin_find(wordt)

        if found or index == -1:
            if not self._add_to_fam(index, wordt):
                return False
        else:
            if not self._insert_to_fam(index, wordt):
                return False
        return True

    def eval_word(self, word: str, save=False) -> (str, float):
        wordt = [word, self._theta(word)]
        found, index = self._bin_find(wordt)

        if found:
            fam = self._family[index][0]
            return fam[0], self._compare(fam, wordt)
        elif save:
            self.add_word(wordt[0])
        return wordt[0], -1

    def pre_process(self, sub_sizes: [int], words: [str]) -> {str, float}:
        self._weights = [1] * (len(sub_sizes) + 1)

        ave_word_len = 0
        unique = []
        counted = []

        for size in sub_sizes:
            self._sub_sizes = [size]
            subs = {}
            for word in words:
                theta = self._theta_prime(word)
                for sub in theta.keys():
                    if sub not in subs:
                        subs[sub] = 0
                    subs[sub] += theta[sub]

                ave_word_len += len(word)
            ave_word_len /= len(words)

            tmp = []
            for sub in subs.keys():
                if sub == '\'': continue
                tmp.append((sub, subs[sub]))

            tot = 0

            _min, _max = 1, 0
            for i in range(len(tmp)):
                sub, val = tmp[i]
                tot += val
            for i in range(len(tmp)):
                sub, val = tmp[i]
                tmp[i] = sub, ((tot-val) / tot)
                _min = min(_min, tmp[i][1])
                _max = max(_max, tmp[i][1])

            print('=== SET {0}'.format(size))
            print('average num splits  = {0}'.format(ave_word_len - size + 1))
            print('total unique subs   = {0}'.format(len(tmp)))
            print('total subs          = {0}\n'.format(tot))
            unique.append(len(tmp))
            counted.append(tot)

            OldRange = _max - _min
            NewRange = 0.99 - 0.01

            for i in range(len(tmp)):
                sub, val = tmp[i]
                tmp[i] = sub, (((val - _min) * NewRange) / OldRange) + 0.01

            subs = sorted(tmp, key=lambda tup: -tup[1])
            del tmp

            f = open('weights_{0}.txt'.format(size), 'w')
            f.write('{\n')
            for i in range(len(subs)):
                sub, val = subs[i]
                f.write('                \'{0}\': {1}{2}\n'.format(sub, val, ',' if i < len(subs) - 1 else ''))
            f.write('            }')
            f.close()

        unique = np.array(unique, dtype=np.float32)
        counted = np.array(counted, dtype=np.float32)
        weights = []

        _min = np.min(unique)
        _max = np.max(unique)
        OldRange = float(_max - _min)
        NewRange = 1.0
        for i in range(len(unique)):
            unique[i] = (1 - float((unique[i] - _min) * NewRange) / OldRange) + 0.1

        _min = np.min(counted)
        _max = np.max(counted)
        OldRange = float(_max - _min)
        NewRange = 0.99
        for i in range(len(counted)):
            counted[i] = (1 - float((counted[i] - _min) * NewRange) / OldRange) + 0.1
            weights.append((unique[i] + counted[i]) * 10)

        print(str(unique))
        print(str(counted))
        print(str(weights))


    def to_string(self):
        s = ''
        num_fam = len(self._family)
        for i in range(num_fam):
            fam = self._family[i]
            s += '{0}: ({1})\n'.format(fam[0][0], i)
            for word in fam[1:]:
                s += '\t{0}\n'.format(word[0])
        return s

    def num_known_words(self):
        return len(self._family)


def make_space(n):
    s = ''
    for i in range(n):
        s += ' '
    return s

if __name__ == '__main__':

    words = [w.strip('\n') for w in open(join('words', 'wordsEn.txt')).readlines() if w != '']
    sorter = Bucket2()

    if False:
        words = [
            'punier',
            'puniest',
            'punily',
            'puniness',
            'punish',
            'punished',
            'punisher',
            'punishers',
            'punishes',
            'punishing',
            'punitive',
            'punk',
            'punker',
            'punner',
            'punners',
            'punnier',
            'punning',
            'punny',
            'puns',
            'punster',
            'punt',
            'punter',
            'punting',
            'punty'
    ]

    random.shuffle(words)
    for word in words:
        sorter.add_word(word)

    print('created {0} word families'.format(sorter.num_known_words()))

    open(join('words', 'WordBank.dat'), 'w').write(sorter.to_string())

    exit(0)

    x = Bucket()
    res = x.step()

    mw = 0
    for sn in res:
        mw = max(mw, len(sn.Word))
        for node in sn.Edges:
            mw = max(mw, len(node.Word))

    s = ''
    for sn in res:
        s += '{0}{1}: '.format(sn.Word, make_space(mw - len(sn.Word)))
        i = 0
        for node in sn.Edges:
            i += 1
            space = make_space(mw - len(node.Word)) + '|'
            s += '{0}{1}'.format(node.Word, space if i < len(sn.Edges) else '')
        s += '\n'
    print(s)
    open('rendered.dat', 'w').write(s)

