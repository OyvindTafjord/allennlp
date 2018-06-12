# Code ported from Peter Clark's Lisp heuristics for "world extraction"

# Light on comments and typing

from collections import defaultdict
import re
import spacy

from allennlp.data.tokenizers.word_stemmer import PorterStemmer


# Various simple helper functions for WorldExtractor below
def get_words(string):
    return re.findall(r'[A-Za-z]+', string)


def words_from_tokens(tokens):
    return [token.text.lower() for token in tokens]


def is_subseq(sequence, subseq):
    sub_len = len(subseq)
    for index in range(0, len(sequence) - sub_len + 1):
        if sequence[index:index+sub_len] == subseq:
            return True
    return False


def get_start_pos(string, search_string):
    pos = string.find(search_string)
    if pos < 0:
        pos = [string.find(word) for word in get_words(search_string)]
        pos = list(filter(lambda x: x >= 0, pos))
        if len(pos) == 0: return -1
        pos = min(pos)
    return pos


def get_words_postag(tokens, postag):
    return [token.text.lower() for token in tokens if token.tag_ == postag]


def is_adjective(token):
    return token.tag_ == "JJ"


def is_noun(token):
    return token.tag_ in ["NNP", "NNS", "NN"]


def starts_simple_np(tokens):
    for token in tokens:
        if is_noun(token): return True
        if not is_adjective(token): return False
    return False


def get_stem_overlaps(query, references, stemmer):
    query_stems = {stemmer.stem(x) for x in get_words(query)}
    references_stems = [{stemmer.stem(x) for x in get_words(reference)} for reference in references]
    return [len(query_stems.intersection(reference_stems)) for reference_stems in references_stems]


def split_question(question):
    return re.split(r' *\([A-F]\) *', question)


class WorldExtractor():
    def __init__(self):
        self._stemmer = PorterStemmer().stemmer
        self._spacy_model = spacy.load('en_core_web_sm')

    def extract(self, question):
        """
        Extract world entities from question using various heuristics. Can return return any
        number of worlds, will typically use the first two
        """
        setup, *answers = split_question(question)
        answers_tokens = [self._spacy_model(answer) for answer in answers]
        answers_words = [words_from_tokens(tokens) for tokens in answers_tokens]
        setup_tokens = self._spacy_model(setup)
        setup_words = words_from_tokens(setup_tokens)

        answer_key_words = []
        # Get non-stop and non-comparative words from answer options
        for answer_words in answers_words:
            res = []
            for word in answer_words:
                tags = word_lookup[word]
                if not 'stop' in tags and not 'comparative' in tags and word.isalpha():
                    res.append(word)
            answer_key_words.append(res)
        worlds = []

        # If there are words left in answer options, try to treat them as worlds
        if min([len(x) for x in answer_key_words]) > 0 and answer_key_words[0] != answer_key_words[1]:
            different_last_words = answer_key_words[0][-1] != answer_key_words[1][-1]
            for answer_index in range(2):
                world = ""
                words = answer_key_words[answer_index]

                # e.g., ("ice") is a world if it occurs in the setup also
                # IF the words in optionA are a subseq of the setup words THEN optionA is the A-WORLD
                if is_subseq(setup_words, words):
                    world = " ".join(words)

                # e.g., A/B ("Joe's" "carpet") vs. ("Joe's" "floor") -> ("carpet") ("floor") worlds
                # ELSE IF the last word in optionA differs from the last word of optionB
                # AND the last word of optionA is also in the setup THEN the last word of optionA is the A-WORLD
                elif words[-1] in setup_words and different_last_words:
                    world = words[-1]

                # e.g., A/B ("red" "carpet") vs. ("green" "carpet") -> ("red") ("green") worlds
                # ELSE IF the first word of optionA is also in the setup THEN the first word of optionA is the A-WORLD
                elif words[0] in setup_words:
                    world = words[0]

                # e.g. A/B ("travelling" "over" "a" "red" "house") vs. ("travelling" "over" "a" "blue" "house") -> ("red") ("blue") worlds
                # ELSE gather just the adjective(s) in optionA. IF the adjective(s) is also in setup, then the adjective(s) is the A-WORLD
                else:
                    adjectives = get_words_postag(answers_tokens[answer_index], "JJ")
                    if any(adjective in setup for adjective in adjectives):
                        world = " ".join(adjectives)
                worlds.append(world)

            # IF found an A-WORLD AND found a B-WORLD AND A-WORLD /= B-WORLD THEN you're done!
            if not "" in worlds and worlds[0] != worlds[1]:
                # Reorder so first world appears first in setup
                setup_pos = [get_start_pos(setup, world) for world in worlds]
                if min(setup_pos) >= 0 and setup_pos[1] < setup_pos[0]:
                    worlds.reverse()
                return worlds
        worlds = []
        tokens = self._spacy_model(question)
        for i in range(len(tokens)):
            word = tokens[i].text

            # e.g., "some grass[world1].... grass" -> "some grass[world1].... grass[world1]"
            # IF word_i was previously seen earlier and tagged as a WORLD, then tag word_i with
            # the already-assigned world number
            if word in worlds:
                continue
            tags_next_word = []
            if i + 1 < len(tokens):
                tags_next_word = word_lookup[tokens[i+1].text]
            tags_word = word_lookup[word]

            # e.g., for "an icy road" or "a gravel road" ->  "an icy[world1] road" or "a gravel[world1] road"
            # and for "a red carpet" -> "a red[world1] carpet"
            # IF word_{i+1} is a surface word
            # AND word_i is a surface-attribute word
            # OR word_i is a surface word
            # OR word_i is an adjective
            # THEN tag word_i as a WORLD - assign it the next world number
            if "surface" in tags_next_word and (
                                "surface" in tags_word or
                                "surface_attribute" in tags_word or
                                is_adjective(tokens[i])):
                worlds.append(word)
                continue
            word_previous = ""
            if i > 0:
                word_previous = tokens[i-1].text

            # e.g., for "a carpet" -> "a carpet[world1]"  BUT BLOCK: "a red[world1] carpet" -> no tagging of carpet
            # IF word_i is a surface word
            # AND word_{i-1} has not already been tagged as a world
            # THEN tag word_i as a WORLD - assign it the next world number
            if "surface" in tags_word and not word_previous in worlds:
                worlds.append(word)
                continue

            # e.g., "over a bridge", "compared with a racetrack" -> "over a bridge[world1]", "compared with a racetrack[world1]
            # IF word_i is a noun (or start of adj+noun noun phrase)
            # AND the nearest, previous non-stop-word is one of ("on" "across" "over" "in" "into" "compared" "through"}
            # THEN tag word_i as a WORLD - assign it the next world number
            if starts_simple_np(tokens[i:]):
                good_preps = ["on", "across", "over", "in", "into", "compared", "through"]
                for j in range(i-1, -1, -1):
                    word_j = tokens[j].text
                    if word_j in good_preps:
                        worlds.append(word)
                        break
                    if not "stop" in word_lookup[word_j]:
                        break
        return worlds

    def align(self, extracted, literals):
        """
        Use stemming to attempt alignment between extracted world and given world literals.
        If more words align to one world vs the other, it's considered aligned.
        """
        literal_keys = list(literals.keys())
        literal_values = list(literals.values())
        overlaps = [get_stem_overlaps(extract, literal_values, self._stemmer) for extract in extracted]
        worlds = []
        for overlap in overlaps:
            if overlap[0] > overlap[1]:
                worlds.append(literal_keys[0])
            elif overlap[0] < overlap[1]:
                worlds.append(literal_keys[1])
            else:
                worlds.append(None)
        return worlds


# Some hardcoded word lists
word_list={}

word_list['surface'] = ["road", "surface", "slope", "track", "tracks", "trail", "floor", "wood", "carpet", "gravel", "soil", "sand", "ice", "grass", "blacktop", "asphalt", "brick", "sandpaper"]

word_list['surface_attribute'] = ["smooth", "rough", "icy", "grassy", "sandy", "soily", "wood", "brick"]

word_list['stop'] = ["aboard", "about", "above", "accordance", "according", "across", "after", "against", "along", "alongside", "amid", "amidst", "apart", "around", "as", "aside", "astride", "at",
                     "atop", "back", "because", "before", "behind", "below", "beneath", "beside", "besides", "between", "beyond", "but", "by", "concerning", "down", "due", "during", "except", "exclusive", "for", "from", "in",
                     "including", "inside", "instead", "into", "irrespective", "less", "minus", "near", "next", "of", "off", "on", "onto", "opposite", "out", "outside", "over", "owing", "per", "prepatory", "previous",
                     "prior", "pursuant", "regarding", "sans", "subsequent", "such", "than", "thanks", "through", "throughout", "thru", "till", "to", "together", "top", "toward", "towards", "under", "underneath",
                     "unlike", "until", "up", "upon", "using", "versus", "via", "with", "within", "without", "out of", "i", "I", "me", "myself", "yourself", "he", "him", "himself", "she", "her", "herself", "it",
                     "itself", "this", "that", "we", "us", "ourselves", "you", "yourselves", "they", "them", "themselves", "these", "those", "mine", "my", "your", "yours", "his", "her", "hers", "its", "our", "ours",
                     "their", "theirs", "who", "what", "where", "which", "why", "when", "how", "how much", "how many", "this", "that", "which", "these", "those", "either", "or", "and", "neither", "the", "an", "a", "am",
                     "are", "is", "was", "were", "will", "be", "if", "then", "not", "do", "also", "true", "false", "by", "of", "as", "but", "for", "be", "occur", "happen", "away", "have", "has", "had"]

word_list['comparative'] = ["further", "farther", "furthest", "smoother", "slicker", "slipperier", "smoothest", "rougher", "roughest", "faster", "fastest", "slower", "slowest", "hotter", "easily", "no", "than",
                            "compared", "most", "largest", "longest", "greatest", "more", "greater", "larger", "bigger", "higher", "longer", "is", "least", "lowest", "smallest", "less", "lesser", "smaller", "lower", "on",
                            "across", "over", "when", "rough", "smooth", "slippery", "slippy", "slick", "far", "way", "mph", "per", "m", "meter", "meters", "feet", "foot", "ft", "inch", "inches", "mile", "miles", "cm",
                            "centimeter", "centimeters", "km", "kilometer", "kilometers", "quickly", "rapidly", "fast", "stop", "stopped", "slowed", "slow", "slowly", "big", "substantial", "long", "huge", "high", "lot",
                            "large", "great", "short", "little", "tiny", "low", "small", "amount", "of", "hot", "speed", "velocity", "friction", "resistance", "heat", "smoke", "temperature", "distance", "rate"]


def reverse_lookup(dict):
    res = defaultdict(list)
    for tag, words in dict.items():
        for word in words:
            res[word.lower()].append(tag)
    return res

# Reverse lookup of word tags
word_lookup = reverse_lookup(word_list)
