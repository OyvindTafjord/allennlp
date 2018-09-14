# Miscellaneous helper functions for friction parser.
# Lots of prototype code

from collections import defaultdict
import re
import spacy

from allennlp.data.tokenizers.word_stemmer import PorterStemmer
#from allennlp.models.archival import load_archive
from allennlp.semparse import util as semparse_util
from allennlp.semparse.worlds import FrictionWorld
#from allennlp.service.predictors import Predictor
from nltk.metrics.distance import edit_distance


LEXICAL_CUES = {}

LEXICAL_CUES["synonyms"] = {
    "friction":["resistance","traction"],
    "speed":["velocity","pace"],
    "distance":["length","way"],
    "heat":["temperature","warmth","smoke"],
    "smoothness":["slickness","roughness"],
    "acceleration":[],
    "amountSweat":["sweat"],
    "apparentSize":["size"],
    "breakability":["brittleness"],
    "brightness":[],
    "exerciseIntensity":["excercise"],
    "flexibility":[],
    "gravity":[],
    "loudness":[],
    "mass":["weight"],
    "strength":["power"],
    "thickness":[],
    "time":[],
    "weight":["mass"]
}

LEXICAL_CUES["values"] = {
    "friction":[],
    "speed":["fast","slow","faster","slower","slowly","quickly","rapidly"],
    "distance":["far","near","further","longer","shorter","long","short","farther","furthest"],
    "heat":["hot","hotter","cold","colder"],
    "smoothness":["rough","smooth","rougher","smoother","bumpy","slicker"],
    "acceleration":[],
    "amountSweat":["sweaty"],
    "apparentSize":["large","small","larger","smaller"],
    "breakability":["brittle","break","solid"],
    "brightness":["bright","shiny","faint"],
    "exerciseIntensity":["run","walk"],
    "flexibility":["flexible","stiff","rigid"],
    "gravity":[],
    "loudness":["loud","faint","louder","fainter"],
    "mass":["heavy","light","heavier","lighter","massive"],
    "strength":["strong","weak","stronger","weaker"],
    "thickness":["thick","thin","thicker","thinner","skinny"],
    "time":["long","short",],
    "weight":["heavy","light","heavier","lighter"]
}


# List of organism and stage entities for life cycle questions
LIFE_CYCLE_ORGANISMS = ["african_goliath_beetle", "bear", "bird", "black_widow", "butterfly",
                        "chicken", "cockroach", "cricket", "crocodile", "darkling_beetle",
                        "dragonfly", "fish", "flea", "frog", "grasshopper", "hammerhead_shark",
                        "hen", "human", "lice", "lizard", "longleaf_pine", "mosquito", "moth",
                        "newt", "oviaporous_snake", "penguin", "praying_mantis", "salmon",
                        "scorpion", "spider", "toad", "tree", "whale", "wolf"]

LIFE_CYCLE_STAGES = ["adolescent", "adult", "adulthood", "alevin", "baby", "big_chick",
                     "bottle_brush", "caterpillar", "chick", "childhood", "cocoon", "cub",
                     "decline", "eft", "egg", "embryo", "fledgling", "foetus", "froglet",
                     "fry", "fully_grown", "gestation", "grass", "hatchling", "juvenile",
                     "larva", "mature", "matured_tree", "nestling", "nymph", "old", "parr",
                     "post_spawn", "prenatal", "pullet", "pup", "pupa", "sapling", "seed",
                     "seedling", "smolt", "snag", "spawner", "spiderling", "sprout", "subadult",
                     "tadpole", "tadpole_with_legs", "teenage", "toadlet", "young", "young_adult"]


# Split entity names into words (camel case, hyphen or underscore)
_regex_decamel = re.compile(r"\B([A-Z])")
def words_from_entity_string(entity):
    res = entity.replace("_", " ").replace("-", " ")
    res = _regex_decamel.sub(r" \1", res)
    return res


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


def nl_triple(triple, nl_world):
    return f"{nl_attr(triple[0]).capitalize()} is {triple[1]} for {nl_world[triple[2]]}"


def nl_arg(arg, nl_world):
    if arg[0] == 'and':
        return [nl_arg(x, nl_world) for x in arg[1:]]
    else:
        return [nl_triple(arg, nl_world)]


def nl_attr(attr):
    return words_from_entity_string(strip_entity_type(attr)).lower()


def nl_dir(sign):
    if sign == 1:
        return "higher"
    else:
        return "lower"

def nl_world_string(world):
    return f'"{str_join(world, "|")}"'


def strip_entity_type(entity):
    return re.sub(r'^[a-z]:', '', entity)


def str_join(string_or_list, joiner, prefixes="", postfixes=""):
    res = string_or_list
    if not isinstance(res, list):
        res = [res]
    res = [f'{prefixes}{x}{postfixes}' for x in res]
    return joiner.join(res)


def get_explanation(logical_form, world_extractions, answer_index, world):
    """
    Create explanation (as a list of header/content entries) for an answer
    """
    output = []
    nl_world = {}
    if world_extractions['world1'] != "N/A" and world_extractions['world1'] != ["N/A"]:
        nl_world['world1'] = nl_world_string(world_extractions['world1'])
        nl_world['world2'] = nl_world_string(world_extractions['world2'])
        output.append({
            "header": "Identified two worlds",
            "content": [f'''world1 = {nl_world['world1']}''',
                        f'''world2 = {nl_world['world2']}''']
        })
    else:
        nl_world['world1'] = 'world1'
        nl_world['world2'] = 'world2'
    parse = semparse_util.lisp_to_nested_expression(logical_form)
    if parse[0][0] != "infer":
        return None
    setup = parse[0][1]
    output.append({
        "header": "The question is stating",
        "content": nl_arg(setup, nl_world)
    })
    answers = parse[0][2:]
    output.append({
        "header": "The answer options are stating",
        "content": ["A: " + " and ".join(nl_arg(answers[0], nl_world)),
                    "B: " + " and ".join(nl_arg(answers[1], nl_world))]
    })
    setup_core = setup
    if setup[0] == 'and':
        setup_core = setup[1]
    s_attr = setup_core[0]
    s_dir = world.qr_size[setup_core[1]]
    s_world = nl_world[setup_core[2]]
    a_attr = answers[answer_index][0]
    qr_dir = world.get_qr_coeff(strip_entity_type(s_attr), strip_entity_type(a_attr))
    a_dir = s_dir * qr_dir
    a_world = nl_world[answers[answer_index][2]]

    content = [f'''When {nl_attr(s_attr)} is {nl_dir(s_dir)} then {nl_attr(a_attr)} is {nl_dir(a_dir)} (for {s_world})''']
    if a_world != s_world:
        content.append(f'''Therefore {nl_attr(a_attr)} is {nl_dir(-a_dir)} for {a_world}''')
    content.append(f"Therefore {chr(65+answer_index)} is the correct answer")

    output.append({
        "header": "Theory used",
        "content": content
    })

    return output


## Code for processing QR specs to/from string format

re_group = re.compile(r"\[([^[\]].*?)\]")
re_sep = re.compile(r" *, *")
re_initletter = re.compile(r" (.)")


def to_camel(string):
    return re_initletter.sub(lambda x: x.group(1).upper(), string)


def from_qr_spec_string(qr_spec):
    res = []
    groups = re_group.findall(qr_spec)
    for group in groups:
        group_split = re_sep.split(group)
        group_dict = {}
        for attribute in group_split:
            sign = 1
            if attribute[0] == "-":
                sign = -1
                attribute = attribute[1:]
            elif attribute[0] == "+":
                attribute = attribute[1:]
            attribute = to_camel(attribute)
            group_dict[attribute] = sign
        res.append(group_dict)
    return res


def to_qr_spec_string(qr_coeff_sets):
    res = []
    signs = {1:"+", -1:"-"}
    for qr_set in qr_coeff_sets:
        first = True
        group_list = []
        for attr, sign in qr_set.items():
            signed_attr = signs[sign] + attr
            if first:
                first = False
                if sign == 1:
                    signed_attr = attr
            group_list.append(signed_attr)
        res.append(f'[{", ".join(group_list)}]')
    return "\n".join(res)


def from_entity_cues_string(cues_string):
    lines = cues_string.split("\n")
    res = {}
    for line in lines:
        line_split = line.split(":")
        head = line_split[0].strip()
        cues = []
        if len(line_split) > 1:
            cues = re_sep.split(line_split[1])
        res[head] = cues
    return res


def from_bio(tags, target):
    res = []
    current = None
    for index, tag in enumerate(tags):
        if tag == "B-" + target:
            if current is not None:
                res.append((current, index))
            current = index
        elif tag == "I-" + target:
            if current is None:
                current = index  # Should not happen
        else:
            if current is not None:
                res.append((current, index))
            current = None
    if current is not None:
        res.append((current, len(tags)))
    return res

def delete_duplicates(e):
    seen = set()
    res = []
    for e1 in e:
        if not e1 in seen:
            seen.add(e1)
            res.append(e1)
    return res

def group_worlds(tags, tokens):
    spans = from_bio(tags, 'world')
    with_strings = [(" ".join(tokens[i:j]), i, j) for i,j in spans]
    with_strings.sort(key=lambda x:len(x[0]), reverse=True)
    substring_groups = []
    ambiguous = []
    for string,i,j in with_strings:
        found = None
        for group_index, group in enumerate(substring_groups):
            for string_g, _, _ in group:
                if string in string_g:
                    if found is None:
                        found = group_index
                    elif found != group_index:
                        found = -1 # Found multiple times
        if found is None:
            substring_groups.append([(string, i, j)])
        elif found >= 0:
            substring_groups[found].append((string, i, j))
        else:
            ambiguous.append((string, i, j))
    nofit = []
    if len(substring_groups) > 2:
        substring_groups.sort(key=len, reverse=True)
        for extra in substring_groups[2:]:
            best_distance = 999
            best_index = None
            string = extra[0][0]  # Use the longest string
            for index, group in enumerate(substring_groups[:2]):
                for string_g, _, _ in group:
                    distance = edit_distance(string_g, string)
                    if distance < best_distance:
                        best_distance = distance
                        best_index = index
            # Heuristics for "close enough"
            if best_index is not None and best_distance < len(string) - 1:
                substring_groups[best_index] += extra
            else:
                nofit.append(extra)
    else:
        substring_groups += [[("N/A", 999, 999)]] * 2   # padding
    substring_groups = substring_groups[:2]
    # Sort by first occurrence
    substring_groups.sort(key = lambda x:min([y[1] for y in x]))
    world_dict = {}
    for index, group in enumerate(substring_groups):
        world_strings = delete_duplicates([x[0] for x in group])
        world_dict['world'+str(index+1)] = world_strings

    return world_dict


class WorldTaggerExtractor:

    def __init__(self, tagger_archive):
        from allennlp.models.archival import load_archive
        from allennlp.service.predictors import Predictor
        self._tagger_archive = load_archive(tagger_archive)
        self._tagger = Predictor.from_archive(self._tagger_archive)

    def get_world_entities(self, question, tokenized_question=None):
        tokenized_question = tokenized_question or self._tagger._dataset_reader._tokenizer.tokenize(question.lower())
        instance = self._tagger._dataset_reader.text_to_instance(question,
                                                                 tokenized_question=tokenized_question)
        output = self._tagger._model.forward_on_instance(instance)
        tokens_text = [t.text for t in tokenized_question]
        res = group_worlds(output['tags'], tokens_text)
        return res


# Code ported from Peter Clark's Lisp heuristics for "world extraction"
class WorldExtractor:
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

        if len(answer_key_words) == 0:
            print(question)
            print(answers)
            print(answers_words)
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
