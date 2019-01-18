# distutils: language = c++
# cython: infer_types=True
# cython: language_level=3
import json
import os
import sys
from collections import Counter

from pkuseg.config import config


class FeatureExtractor:

    keywords = "-._,|/*:"

    num = set("0123456789." "几二三四五六七八九十千万亿兆零" "１２３４５６７８９０％")
    letter = set(
        "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ" "ａｂｃｄｅｆｇｈｉｇｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ" "／・－"
    )

    keywords_translate_table = str.maketrans("-._,|/*:", "&&&&&&&&")

    @classmethod
    def keyword_rename(cls, text):
        return text.translate(cls.keywords_translate_table)

    @classmethod
    def _num_letter_normalize_char(cls, character):
        if character in cls.num:
            return "**Num"
        if character in cls.letter:
            return "**Letter"
        return character

    @classmethod
    def normalize_text(cls, text):
        text = cls.keyword_rename(text)
        for character in text:
            if config.numLetterNorm:
                yield cls._num_letter_normalize_char(character)
            else:
                yield character

    @staticmethod
    def get_slice_str(iterable, start, length):
        all_len = len(iterable)
        if start < 0 or start >= all_len:
            return ""
        if start + length >= all_len + 1:
            return ""
        return "".join(iterable[start : start + length])

    def __init__(self):

        self.unigram = set()  # type: Set[str]
        self.bigram = set()  # type: Set[str]
        self.feature_to_idx = {}  # type: Dict[str, int]
        self.tag_to_idx = {}  # type: Dict[str, int]

    def build(self, train_file):
        with open(train_file, "r", encoding="utf8") as reader:
            lines = reader.readlines()

        examples = []  # type: List[List[List[str]]]

        # first pass to collect unigram and bigram and tag info
        word_length_info = Counter()
        specials = set()
        for line in lines:
            line = line.strip("\n\r")  # .replace("\t", " ")
            if not line:
                continue

            line = self.keyword_rename(line)

            # str.split() without sep sees consecutive whiltespaces as one separator
            # e.g., '\ra \t　b \r\n'.split() = ['a', 'b']
            words = [word for word in line.split()]

            word_length_info.update(map(len, words))
            specials.update(word for word in words if len(word)>=10)
            self.unigram.update(words)

            for pre, suf in zip(words[:-1], words[1:]):
                self.bigram.add("{}*{}".format(pre, suf))

            example = [
                self._num_letter_normalize_char(character)
                for word in words
                for character in word
            ]
            examples.append(example)

        max_word_length = max(word_length_info.keys())
        for length in range(1, max_word_length + 1):
            print("length = {} : {}".format(length, word_length_info[length]))
        print('special words: {}'.format(', '.join(specials)))
        # second pass to get features

        feature_freq = Counter()

        for example in examples:
            for i, _ in enumerate(example):
                node_features = self.get_node_features(i, example)
                feature_freq.update(
                    feature for feature in node_features if feature != "/"
                )

        feature_set = (
            feature
            for feature, freq in feature_freq.most_common()
            if freq > config.featureTrim
        )
        self.feature_to_idx = {
            feature: idx for idx, feature in enumerate(feature_set)
        }

        if config.nLabel == 2:
            B = B_single = "B"
            I_first = I = I_end = "I"
        elif config.nLabel == 3:
            B = B_single = "B"
            I_first = I = "I"
            I_end = "I_end"
        elif config.nLabel == 4:
            B = "B"
            B_single = "B_single"
            I_first = I = "I"
            I_end = "I_end"
        elif config.nLabel == 5:
            B = "B"
            B_single = "B_single"
            I_first = "I_first"
            I = "I"
            I_end = "I_end"

        tag_set = {B, B_single, I_first, I, I_end}
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(sorted(tag_set))}

    def get_node_features(self, idx, wordary):
        flist = []
        w = wordary[idx]

        # 1 start feature
        flist.append("$$")

        # 8 unigram/bgiram feature
        flist.append("c." + w)
        if idx > 0:
            flist.append("c-1." + wordary[idx - 1])
        else:
            flist.append("/")
        if idx < len(wordary) - 1:
            flist.append("c1." + wordary[idx + 1])
        else:
            flist.append("/")
        if idx > 1:
            flist.append("c-2." + wordary[idx - 2])
        else:
            flist.append("/")
        if idx < len(wordary) - 2:
            flist.append("c2." + wordary[idx + 2])
        else:
            flist.append("/")
        if idx > 0:
            flist.append("c-1c." + wordary[idx - 1] + config.delimInFeature + w)
        else:
            flist.append("/")
        if idx < len(wordary) - 1:
            flist.append("cc1." + w + config.delimInFeature + wordary[idx + 1])
        else:
            flist.append("/")
        if idx > 1:
            flist.append(
                "c-2c-1."
                + wordary[idx - 2]
                + config.delimInFeature
                + wordary[idx - 1]
            )
        else:
            flist.append("/")

        # no num/letter based features
        if not config.wordFeature:
            return flist

        # 2 * (wordMax-wordMin+1) word features (default: 2*(6-2+1)=10 )
        # the character starts or ends a word
        tmplst = []
        for l in range(config.wordMax, config.wordMin - 1, -1):
            # length 6 ... 2 (default)
            # "prefix including current c" wordary[n-l+1, n]
            # current character ends word
            tmp = self.get_slice_str(wordary, idx - l + 1, l)
            if tmp != "":
                if tmp in self.unigram:
                    flist.append("w-1." + tmp)
                    tmplst.append(tmp)
                else:
                    flist.append("/")
                    tmplst.append("**noWord")
            else:
                flist.append("/")
                tmplst.append("**noWord")
        prelst_in = tmplst

        tmplst = []
        for l in range(config.wordMax, config.wordMin - 1, -1):
            # "suffix" wordary[n, n+l-1]
            # current character starts word
            tmp = self.get_slice_str(wordary, idx, l)
            if tmp != "":
                if tmp in self.unigram:
                    flist.append("w1." + tmp)
                    tmplst.append(tmp)
                else:
                    flist.append("/")
                    tmplst.append("**noWord")
            else:
                flist.append("/")
                tmplst.append("**noWord")
        postlst_in = tmplst

        # these are not in feature list
        tmplst = []
        for l in range(config.wordMax, config.wordMin - 1, -1):
            # "prefix excluding current c" wordary[n-l, n-1]
            tmp = self.get_slice_str(wordary, idx - l, l)
            if tmp != "":
                if tmp in self.unigram:
                    tmplst.append(tmp)
                else:
                    tmplst.append("**noWord")
            else:
                tmplst.append("**noWord")
        prelst_ex = tmplst

        tmplst = []
        for l in range(config.wordMax, config.wordMin - 1, -1):
            # "suffix excluding current c" wordary[n+1, n+l]
            tmp = self.get_slice_str(wordary, idx + 1, l)
            if tmp != "":
                if tmp in self.unigram:
                    tmplst.append(tmp)
                else:
                    tmplst.append("**noWord")
            else:
                tmplst.append("**noWord")
        postlst_ex = tmplst

        # this character is in the middle of a word
        # 2*(wordMax-wordMin+1)^2 (default: 2*(6-2+1)^2=50)

        for pre in prelst_ex:
            for post in postlst_in:
                bigram = pre + "*" + post
                if bigram in self.bigram:
                    flist.append("ww.l." + bigram)
                else:
                    flist.append("/")

        for pre in prelst_in:
            for post in postlst_ex:
                bigram = pre + "*" + post
                if bigram in self.bigram:
                    flist.append("ww.r." + bigram)
                else:
                    flist.append("/")

        return flist

    def convert_feature_file_to_idx_file(
        self, feature_file, feature_idx_file, tag_idx_file
    ):

        with open(feature_file, "r", encoding="utf8") as reader:
            lines = reader.readlines()

        with open(feature_idx_file, "w", encoding="utf8") as f_writer, open(
            tag_idx_file, "w", encoding="utf8"
        ) as t_writer:

            f_writer.write("{}\n\n".format(len(self.feature_to_idx)))
            t_writer.write("{}\n\n".format(len(self.tag_to_idx)))

            tags_idx = []  # type: List[str]
            features_idx = []  # type: List[List[str]]
            for line in lines:
                line = line.strip()
                if not line:
                    # sentence finish
                    for feautre_idx in features_idx:
                        if not features_idx:
                            f_writer.write("0\n")
                        else:
                            f_writer.write(",".join(map(str, feautre_idx)))
                            f_writer.write("\n")
                    f_writer.write("\n")

                    t_writer.write(",".join(map(str, tags_idx)))
                    t_writer.write("\n\n")

                    tags_idx = []
                    features_idx = []
                    continue

                splits = line.split(" ")
                feature_idx = [
                    self.feature_to_idx[feat]
                    for feat in splits[1:-1]
                    if feat in self.feature_to_idx
                ]
                features_idx.append(feature_idx)
                tags_idx.append(self.tag_to_idx[splits[-1]])

    def convert_text_file_to_feature_file(
        self, text_file, conll_file=None, feature_file=None
    ):

        if conll_file is None:
            conll_file = "{}.conll{}".format(*os.path.split(text_file))
        if feature_file is None:
            feature_file = "{}.feat{}".format(*os.path.split(text_file))

        if config.nLabel == 2:
            B = B_single = "B"
            I_first = I = I_end = "I"
        elif config.nLabel == 3:
            B = B_single = "B"
            I_first = I = "I"
            I_end = "I_end"
        elif config.nLabel == 4:
            B = "B"
            B_single = "B_single"
            I_first = I = "I"
            I_end = "I_end"
        elif config.nLabel == 5:
            B = "B"
            B_single = "B_single"
            I_first = "I_first"
            I = "I"
            I_end = "I_end"

        conll_line_format = "{} {}\n"

        with open(text_file, "r", encoding="utf8") as reader, open(
            conll_file, "w", encoding="utf8"
        ) as c_writer, open(feature_file, "w", encoding="utf8") as f_writer:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                words = self.keyword_rename(line).split()
                example = []
                tags = []
                for word in words:
                    word_length = len(word)
                    for idx, character in enumerate(word):
                        if word_length == 1:
                            tag = B_single
                        elif idx == 0:
                            tag = B
                        elif idx == word_length - 1:
                            tag = I_end
                        elif idx == 1:
                            tag = I_first
                        else:
                            tag = I
                        c_writer.write(conll_line_format.format(character, tag))

                        if config.numLetterNorm:
                            example.append(
                                self._num_letter_normalize_char(character)
                            )
                        else:
                            example.append(character)
                        tags.append(tag)
                c_writer.write("\n")

                for idx, tag in enumerate(tags):
                    features = self.get_node_features(idx, example)
                    features = [
                        (feature if feature in self.feature_to_idx else "/")
                        for feature in features
                    ]
                    features.append(tag)
                    f_writer.write(" ".join(features))
                    f_writer.write("\n")
                f_writer.write("\n")

    def save(self):
        data = {}
        data["unigram"] = sorted(list(self.unigram))
        data["bigram"] = sorted(list(self.bigram))
        data["feature_to_idx"] = self.feature_to_idx
        data["tag_to_idx"] = self.tag_to_idx

        with open(
            os.path.join(config.modelDir, "features.json"), "w", encoding="utf8"
        ) as writer:
            json.dump(data, writer, ensure_ascii=False)

    @classmethod
    def load(cls):
        extractor = cls.__new__(cls)
        feature_path = os.path.join(config.modelDir, "features.json")
        if os.path.exists(feature_path):
            with open(feature_path, "r", encoding="utf8") as reader:
                data = json.load(reader)
            extractor.unigram = set(data["unigram"])
            extractor.bigram = set(data["bigram"])
            extractor.feature_to_idx = data["feature_to_idx"]
            extractor.tag_to_idx = data["tag_to_idx"]

            return extractor
        print(
            "WARNING: features.json does not exist, try loading using old format",
            file=sys.stderr,
        )

        with open(
            os.path.join(config.modelDir, "unigram_word.txt"),
            "r",
            encoding="utf8",
        ) as reader:
            extractor.unigram = set([line.strip() for line in reader])

        with open(
            os.path.join(config.modelDir, "bigram_word.txt"),
            "r",
            encoding="utf8",
        ) as reader:
            extractor.bigram = set(line.strip() for line in reader)

        extractor.feature_to_idx = {}
        feature_base_name = os.path.join(config.modelDir, "featureIndex.txt")
        for i in range(10):
            with open(
                "{}_{}".format(feature_base_name, i), "r", encoding="utf8"
            ) as reader:
                for line in reader:
                    feature, index = line.split(" ")
                    feature = ".".join(feature.split(".")[1:])
                    extractor.feature_to_idx[feature] = int(index)

        extractor.tag_to_idx = {}
        with open(
            os.path.join(config.modelDir, "tagIndex.txt"), "r", encoding="utf8"
        ) as reader:
            for line in reader:
                tag, index = line.split(" ")
                extractor.tag_to_idx[tag] = int(index)

        print(
            "INFO: features.json is saved",
            file=sys.stderr,
        )
        extractor.save()

        return extractor
