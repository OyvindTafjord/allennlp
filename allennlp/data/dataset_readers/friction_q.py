"""
Reader for Friction questions
"""

from typing import Any, Dict, List, Optional, Union
import json
import logging
import re

from copy import deepcopy
import numpy as np
from overrides import overrides
import random

import tqdm

from allennlp.common import Params
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_stemmer import PorterStemmer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import ArrayField, Field, TextField, KnowledgeGraphField, LabelField
from allennlp.data.fields import IndexField, ListField, MetadataField, ProductionRuleField
from allennlp.data.fields import SequenceLabelField
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.seq2seq import START_SYMBOL, END_SYMBOL
from allennlp.semparse.contexts import TableQuestionKnowledgeGraph
from allennlp.semparse.contexts.knowledge_graph import KnowledgeGraph
from allennlp.semparse.friction_q_util import WorldExtractor, LEXICAL_CUES, words_from_entity_string
from allennlp.semparse.friction_q_util import LIFE_CYCLE_ORGANISMS, LIFE_CYCLE_STAGES
from allennlp.semparse.worlds import FrictionWorld


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("friction_q")
class FrictionQDatasetReader(DatasetReader):
    """
    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.
    filter_type2 : ``bool`` (optional, default=False)
        Remove type 2 logical forms from the instances, based on presence of "(and" in LF
    first_lf_only : ``bool`` (optional, default=False)
        Use only the first given LF
    split_question : ``bool`` (optional, default=False)
        Split question and LFs to each answer option
    replace_blanks : ``bool`` (optional, default=False)
        When split_question is True, this option will replace ____ in question by the answer option
    flip_answers : ``bool`` (optional, default=False)
        If true, doubles the dataset by flipping the answer options
    flip_worlds : ``bool`` (optional, default=False)
        If true, doubles the dataset by flipping world1 <-> world2
    use_extracted_world_entities : ``bool`` (optional, default=False)
        Use preprocessed world entity strings as part of the LFs
    replace_world_entities : ``bool`` (optional, default=False)
        Replace world entities (w/stemming) with "worldone" and "worldtwo" directly in the question
    extract_world_entities : ``bool`` (optional, default=False)
        Extract world entities using heuristics and align to annotated world literals
    skip_world_alignment : ``bool`` (optional, default=False)
        Don't attempt to align world entities to literals, thus use both LFs
    single_lf_extractor_aligned : ``bool`` (optional, default=False)
        Use the gold LF which is aligned to the world order given by extractor
    gold_worlds : ``bool`` (optional, default=False)
        Use gold worlds rather than extraction heuristics
    tagger_only : ``bool`` (optional, default=False)
        Only output tagging information, in format for CRF tagger
    collapse_tags : ``bool`` (optional, default=False)
        For tagging purposes, collapse classes of tags (like 'world' or 'comparison') to single
        tag (but separate BIO)
    entity_tag_mode : ``str`` (optional, default=None)
        If set, add a field for entity tags ("simple" = 1.0 value for world1 and world2,
        "simple_collapsed" = single 1.0 value for any world), tagging based on token matches
        with extracted entities
    lf_syntax: ``str``
        Which LF formalism to use, see friction types for details

    """
    def __init__(self,
                 lazy: bool = False,
                 filter_type2: bool = False,
                 first_lf_only: bool = False,
                 sample: int = -1,
                 split_question: bool = False,
                 replace_blanks: bool = False,
                 flip_answers: bool = False,
                 flip_worlds: bool = False,
                 lf_syntax: str = None,
                 use_extracted_world_entities: bool = False,
                 extract_world_entities: bool = False,
                 replace_world_entities: bool = False,
                 skip_world_alignment: bool = False,
                 single_lf_extractor_aligned: bool = False,
                 gold_worlds: bool = False,
                 tagger_only: bool = False,
                 collapse_tags: Optional[List[str]] = None,
                 denotation_only: bool = False,
                 skip_attributes_regex: Optional[str] = None,
                 entity_tag_mode: Optional[str] = None,
                 entity_types: Optional[List[str]] = None,
                 lexical_cues: List[str] = None,
                 tokenizer: Tokenizer = None,
                 question_token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._question_token_indexers = question_token_indexers or {"tokens": SingleIdTokenIndexer()}
        # Fake table so we can use WikiTable parser model
        self._knowledge_graph = TableQuestionKnowledgeGraph.read_from_json(
            {"columns":["foo"], "cells": [["foo"]], "question":[]})
        self._world = FrictionWorld(self._knowledge_graph, lf_syntax)
        self._table_token_indexers = self._question_token_indexers
        self._filter_type2 = filter_type2
        self._first_lf_only = first_lf_only
        self._sample = sample
        self._split_question = split_question
        self._replace_blanks = replace_blanks
        self._flip_answers = flip_answers
        self._flip_worlds = flip_worlds
        self._use_extracted_world_entities = use_extracted_world_entities
        self._replace_world_entities = replace_world_entities
        self._skip_world_alignment = skip_world_alignment
        self._lf_syntax = lf_syntax
        self._entity_tag_mode = entity_tag_mode
        self._extract_world_entities = extract_world_entities
        self._single_lf_extractor_aligned = single_lf_extractor_aligned
        self._gold_worlds = gold_worlds
        self._entity_types = entity_types
        self._tagger_only = tagger_only
        self._collapse_tags = collapse_tags
        self._denotation_only = denotation_only
        self._skip_attributes_regex = None
        if skip_attributes_regex is not None:
            self._skip_attributes_regex = re.compile(skip_attributes_regex)
        self._lexical_cues = lexical_cues

        all_entities = {}
        all_entities["comparison"] = ["distance-higher", "distance-lower", "friction-higher",
                            "friction-lower", "heat-higher", "heat-lower", "smoothness-higher",
                            "smoothness-lower", "speed-higher", "speed-lower"]
        all_entities["value"] = ["distance-high", "distance-low", "friction-high", "friction-low",
                                  "heat-high", "heat-low", "smoothness-high", "smoothness-low",
                                  "speed-high", "speed-low"]
        all_entities["world"] = ["world1", "world2"]
        all_entities["vehicle"] = ["vehicle"]

        self._all_entities = None
        # entity_tag_mode:
        #  collapsed = collapse worlds
        #  collapsed2 = collapse all
        #  collapsed3 = collapse all but worlds
        if entity_types is not None:
            if "collapsed2" in self._entity_tag_mode:
                self._all_entities = entity_types
            elif "collapsed3" in self._entity_tag_mode:
                self._all_entities = entity_types
                if "world" in entity_types:
                    self._all_entities.remove("world")
                    self._all_entities += all_entities['world']
            else:
                self._all_entities = [e for t in entity_types for e in all_entities[t]]
                if "collapsed" in self._entity_tag_mode and 'world1' in self._all_entities:
                    self._all_entities.remove('world1')
                    self._all_entities.remove('world2')
                    self._all_entities.append('world')
        logger.info("ALL ENTITIES = {}".format(self._all_entities))

        self._dynamic_entities = dict()
        self._use_attr_entities = False
        if "_attr_entities" in lf_syntax:
            self._use_attr_entities = True
            qr_coeff_sets = self._world.qr_coeff_sets
            if "_general" not in lf_syntax:
                qr_coeff_sets = qr_coeff_sets[:1]
            for s in qr_coeff_sets:
                for k in s.keys():
                    if (self._skip_attributes_regex is not None and
                            self._skip_attributes_regex.search(k)):
                        continue
                    entity_strings = [words_from_entity_string(k.lower())]
                    if self._lexical_cues is not None:
                        for key in self._lexical_cues:
                            if k in LEXICAL_CUES[key]:
                                entity_strings += LEXICAL_CUES[key][k]
                    self._dynamic_entities["a:"+k] = " ".join(entity_strings)
            logger.info(f"DYNAMIC ENTITIES = {self._dynamic_entities}")

        if lf_syntax == "life_cycle_entities":
            self._use_attr_entities = True
            for a in LIFE_CYCLE_ORGANISMS:
                self._dynamic_entities["o:"+a] = a.replace("_", " ")
            for s in LIFE_CYCLE_STAGES:
                self._dynamic_entities["s:"+s] = s.replace("_", " ")


        if self._use_attr_entities:
            logger.info(f"DYNAMIC ENTITIES = {self._dynamic_entities}")
            neighbors = {key: [] for key in self._dynamic_entities.keys()}
            self._knowledge_graph = KnowledgeGraph(entities=set(self._dynamic_entities.keys()),
                                             neighbors=neighbors,
                                             entity_text=self._dynamic_entities)
            self._world = FrictionWorld(self._knowledge_graph, self._lf_syntax)




        self._random_lf_pick = False

        if self._replace_world_entities:
            self._stemmer = PorterStemmer().stemmer

        if self._extract_world_entities:
            self._world_extractor = WorldExtractor()


    @overrides
    def _read(self, file_path):
        debug = 5
        counter = self._sample
        attr_regex = re.compile(r"""\((\w+) (high|low|higher|lower)""")
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file: %s", file_path)
            for line in tqdm.tqdm(data_file):
                counter -= 1
                if counter == 0:
                    break
                line = line.strip("\n")
                if not line:
                    continue
                question_data_in = json.loads(line)
                if self._gold_worlds and 'world_literals' in question_data_in:
                    question_data_in['world_extractions'] = question_data_in['world_literals']
                if self._use_attr_entities:
                    # Add "a:" prefix to attributes in LF
                    lfs = question_data_in['logical_forms']
                    lfs_new = [attr_regex.sub(r"(a:\1 \2", lf) for lf in lfs]
                    question_data_in['logical_forms'] = lfs_new
                world_flip_lf = False
                if self._extract_world_entities:
                    extractions, world_flip_lf = self.get_world_extractions(question_data_in)
                    if extractions is None:
                        continue
                    question_data_in['world_extractions'] = extractions
                if self._entity_types is not None:
                    question_data_in['entity_literals'] = \
                        self._get_entity_literals(question_data_in)

                # Skip training instances without world entities if needing them
                if (self._use_extracted_world_entities or self._replace_world_entities) \
                        and "world_extractions" not in question_data_in \
                        and (self._entity_tag_mode is None or "label" not in self._entity_tag_mode):
                    continue
                if self._split_question:
                    question_datas = self._split_instance(question_data_in, self._replace_blanks)
                elif self._flip_answers:
                    question_datas = self._do_flip_answers(question_data_in)
                elif self._flip_worlds:
                    question_datas = self._do_flip_worlds(question_data_in)
                    # question_datas = [question_datas[1]]
                    # question_datas = [random.choice(question_datas)]
                else:
                    question_datas = [question_data_in]

                if self._replace_world_entities:
                    question_datas = [self._replace_stemmed_entities(data, self._stemmer) for
                                      data in question_datas]
                debug -= 1
                if debug > 0:
                    logger.info(question_datas)
                for question_data in question_datas:
                    question = question_data['question']
                    question = self._fix_question(question)
                    question_id = question_data['id']
                    logical_forms = question_data['logical_forms']
                    if (self._skip_attributes_regex is not None and
                            self._skip_attributes_regex.search(logical_forms[0])):
                        continue
                    one_lf_only = self._first_lf_only
                    if self._single_lf_extractor_aligned:
                        if "world_literals" in question_data_in:
                            one_lf_only = True
                    elif (self._use_extracted_world_entities or
                              self._replace_world_entities) and not self._skip_world_alignment:
                        one_lf_only = True
                    if one_lf_only and len(logical_forms) > 1:
                        if self._random_lf_pick:
                            logical_forms = [random.choice(logical_forms)]
                        elif world_flip_lf:
                            logical_forms = [logical_forms[1]]
                        else:
                            logical_forms = [logical_forms[0]]
                    if debug > 0:
                        logger.info(logical_forms)
                    # Hacky filter to ignore "type2" questions
                    if self._filter_type2 and len(logical_forms) > 0 and "(and " in logical_forms[0]:
                        continue
                    answer_index = question_data['answer_index']
                    world_extractions = question_data.get('world_extractions')
                    entity_literals = question_data.get('entity_literals')
                    if entity_literals is not None and world_extractions is not None:
                        # This will catch flipped worlds if need be
                        entity_literals.update(world_extractions)
                    additional_metadata = {'id': question_id,
                                           'question': question,
                                           'answer_index': answer_index,
                                           'logical_forms': logical_forms}

                    yield self.text_to_instance(question, logical_forms,
                                                    additional_metadata, world_extractions,
                                                    entity_literals, debug=debug)

    @overrides
    def text_to_instance(self,  # type: ignore
                         question: str,
                         logical_forms: List[str] = None,
                         additional_metadata: Dict[str, Any] = None,
                         world_extractions: Dict[str, str] = None,
                         entity_literals: Dict[str, Union[str, List[str]]] = None,
                         tokenized_question: List[Token] = None, debug=None) -> Instance:
        """

        """
        # pylint: disable=arguments-differ
        tokenized_question = tokenized_question or self._tokenizer.tokenize(question.lower())
        additional_metadata = additional_metadata or dict()
        additional_metadata['question_tokens'] = [token.text for token in tokenized_question]
        if world_extractions is not None:
            additional_metadata['world_extractions'] = world_extractions
        question_field = TextField(tokenized_question, self._question_token_indexers)
        if self._use_extracted_world_entities and world_extractions is not None:
            neighbors = {key: [] for key in world_extractions.keys()}
            knowledge_graph = KnowledgeGraph(entities=set(world_extractions.keys()),
                                             neighbors=neighbors,
                                             entity_text=world_extractions)
            world = FrictionWorld(knowledge_graph, self._lf_syntax)
        else:
            knowledge_graph = self._knowledge_graph
            world = self._world

        table_field = KnowledgeGraphField(knowledge_graph,
                                          tokenized_question,
                                          self._table_token_indexers,
                                          tokenizer=self._tokenizer)

        # print(table_field._compute_linking_features())
        world_field = MetadataField(world)

        production_rule_fields: List[Field] = []
        for production_rule in world.all_possible_actions():
            _, rule_right_side = production_rule.split(' -> ')
            is_global_rule = not world.is_table_entity(rule_right_side)
            field = ProductionRuleField(production_rule, is_global_rule)
            production_rule_fields.append(field)
        action_field = ListField(production_rule_fields)

        fields = {'question': question_field,
                  'table': table_field,
                  'world': world_field,
                  'actions': action_field}

        if self._denotation_only:
            denotation_field = LabelField(additional_metadata['answer_index'], skip_indexing=True)
            fields['denotation_target'] = denotation_field

        if self._entity_types is not None and entity_literals is not None:
            entity_tags = self._get_entity_tags_full(table_field, entity_literals, tokenized_question)
            if debug > 0:
                logger.info(entity_literals)
                logger.info(self._all_entities)
                logger.info(entity_tags)
            if self._tagger_only:
                entity_tags = self._convert_tags_bio(entity_tags)
                fields = {'tokens': question_field}
                fields['tags'] = SequenceLabelField(entity_tags, question_field)
                additional_metadata['tags_gold'] = entity_tags
                fields['metadata'] = MetadataField(additional_metadata)
                return Instance(fields)

            elif self._entity_tag_mode == "label":
                fields['target_entity_tag'] = SequenceLabelField(entity_tags, question_field)
            else:
                # Convert to one-hot
                additional_metadata['entity_tags'] = entity_tags
                entity_tags = np.eye(len(self._all_entities)+1)[entity_tags]
                fields['entity_tag'] = ArrayField(entity_tags)

        elif self._entity_tag_mode is not None and world_extractions is not None:
            linking_features = table_field.linking_features
            entity_tags = self._get_entity_tags(linking_features)
            if self._entity_tag_mode == "simple":
                entity_tags = [[[0,0], [1,0], [0,1]][tag] for tag in entity_tags]
            elif self._entity_tag_mode == "simple_collapsed":
                entity_tags = [[[0], [1], [1]][tag] for tag in entity_tags]
            elif self._entity_tag_mode == "simple3":
                entity_tags = [[[1,0,0], [0,1,0], [0,0,1]][tag] for tag in entity_tags]

            if self._entity_tag_mode == "label":
                fields['target_entity_tag'] = SequenceLabelField(entity_tags, question_field)
            else:
                entity_tag_field = ArrayField(np.array(entity_tags))
                fields['entity_tag'] = entity_tag_field

        if logical_forms:
            action_map = {action.rule: i for i, action in enumerate(action_field.field_list)}
            action_sequence_fields: List[Field] = []
            for logical_form in logical_forms:
                expression = world.parse_logical_form(logical_form)
                action_sequence = world.get_action_sequence(expression)
                try:
                    index_fields: List[Field] = []
                    for production_rule in action_sequence:
                        index_fields.append(IndexField(action_map[production_rule], action_field))
                    action_sequence_fields.append(ListField(index_fields))
                except KeyError as error:
                    logger.info(f'Missing production rule: {error.args}, skipping logical form')
                    logger.info(f'Question was: {question}')
                    logger.info(f'Logical form was: {logical_form}')
                    continue
            fields['target_action_sequences'] = ListField(action_sequence_fields)
        fields['metadata'] = MetadataField(additional_metadata or {})
        return Instance(fields)

    def _convert_tags_bio(self, tags):
        res = []
        last_tag = 0
        prefix_i = "I-"
        prefix_b = "B-"
        all_tags = self._all_entities
        if self._collapse_tags is not None:
            if 'world' in self._collapse_tags:
                all_tags = ['world' if 'world' in x else x for x in all_tags]
            if 'comparison' in self._collapse_tags:
                all_tags = ['comparison' if '-higher' in x or '-lower' in x else x for x in all_tags]
            if 'value' in self._collapse_tags:
                all_tags = ['value' if '-high' in x or '-low' in x else x for x in all_tags]
        if self._entity_tag_mode == "label":
            prefix_i = ""
            prefix_b = ""
        for tag in tags:
            if tag == 0:
                bio_tag = "O"
            elif tag == last_tag:
                bio_tag = prefix_i + all_tags[tag-1]
            else:
                bio_tag = prefix_b + all_tags[tag-1]
            last_tag = tag
            res.append(bio_tag)
        return res



    @staticmethod
    def _fix_question(question: str) -> str:
        """
        Replace answer dividers (A), (B) etc with a unique token answeroptionA, answeroptionB, ...
        Replace '_____' with 'blankblank'
        """
        res = re.sub(r'\(([A-G])\)', r"answeroption\1", question)
        res = re.sub(r" *_{3,} *", " blankblank ", res)
        return res


    def get_world_extractions(self, question_data):
        extracted = self._world_extractor.extract(question_data['question'])
        extractions = {}
        flip = False
        if 'world_literals' in question_data and not self._skip_world_alignment:
            literals = question_data['world_literals']
            aligned = self._world_extractor.align(extracted, literals)
            # If we haven't aligned two different things (including None), give up
            if len(set(aligned)) < 2:
                return None, flip
            aligned_dict = {key: value for key, value in zip(aligned, extracted)}
            for key in literals.keys():
                # if key is missing, then it must be assigned to None per above logic
                value = aligned_dict[key] if key in aligned_dict else aligned_dict[None]
                extractions[key] = value
            if self._single_lf_extractor_aligned:
                if extractions["world1"] != extracted[0]:
                    flip = True
                extractions = {"world1": extracted[0], "world2": extracted[1]}
        else:
            if len(extracted) < 2 or extracted[0] == extracted[1]:
                return None, flip
            extractions = {"world1": extracted[0], "world2": extracted[1]}
            # extractions = {"world1": extracted[1], "world2": extracted[0]}
        return extractions, flip

    def _get_entity_tags_full(self, table_field, entity_literals, tokenized_question):
        res = []
        features = table_field._feature_extractors[8:]
        for i, token in enumerate(tokenized_question):
            tag_best = 0
            score_max = 0
            for tag_index, tag in enumerate(self._all_entities):
                literals = entity_literals.get(tag, [])
                if not isinstance(literals, list):
                    literals = [literals]
                for literal in literals:
                    tag_tokens = self._tokenizer.tokenize(literal.lower())
                    scores = [fe(tag, tag_tokens, token, i, tokenized_question) for fe in features]
                    # Small tie breaker in favor of longer sequences
                    score = max(scores) + len(tag_tokens)/100
                    if score > score_max and score >= 0.5:
                        tag_best = tag_index + 1
                        score_max = score
            res.append(tag_best)
        return res

    def _get_entity_literals(self, question_data):
        res = {}
        make_list = lambda x: x if isinstance(x, list) else [x]
        for key, value in question_data.items():
            if '_literals' in key and key.replace('_literals', '') in self._entity_types:
                if "collapsed2" in self._entity_tag_mode \
                        or ("collapsed3" in self._entity_tag_mode and 'world' not in key):
                    tag = key.replace('_literals', '')
                    values = []
                    for v in value.values():
                        values += make_list(v)
                    res[tag] = values
                else:
                    res.update(value)
        if "collapsed" in self._entity_tag_mode and "collapsed3" not in self._entity_tag_mode and "world1" in res:
            worlds = []
            for world in ['world1', 'world2']:
                if world in res:
                    worlds = worlds + make_list(res[world])
                    del res[world]
            res['world'] = worlds
        return res

    @staticmethod
    def _get_entity_tags(linking_features: List[List[List[float]]]) -> List[int]:
        """
        Simple heuristic on linking features to get entity tag (0=no world, 1=world1, 2=world2)
        """
        res = []
        for token_index, entity_features in enumerate(zip(*linking_features)):
            score_max = 0
            world = 0
            for index, features in enumerate(entity_features):
                # Use the two span features with cutoff of 0.5
                score = max(features[8:])
                if score > score_max and score >= 0.5:
                    world = index + 1
                    score_max = score
            res.append(world)
        return res

    @staticmethod
    def _make_action_sequence_field(action_sequence: List[str]) -> ListField:
        action_sequence.insert(0, START_SYMBOL)
        action_sequence.append(END_SYMBOL)
        return ListField([LabelField(action, label_namespace='actions') for action in action_sequence])

    @staticmethod
    def _hacky_split_lf(lf):
        regex = re.compile(r'^(.*\) )(\([^)]+\)) (\([^)]+\))\)$')
        match = regex.match(lf)
        return match.groups()

    @staticmethod
    def _split_instance(json, replace_blanks):
        split_q = re.split(r' *\([A-F]\) *', json['question'])
        split_lfs = [FrictionQDatasetReader._hacky_split_lf(lf) for lf in json['logical_forms']]
        answer_index = json['answer_index']
        id_ = json['id'] + "_"
        q_core = split_q[0]
        res = []
        for index in [0,1]:
            answer = split_q[index+1]
            if replace_blanks and '___' in q_core:
                question = re.sub(r' *_{3,} *', " " + answer + " ", q_core)
            else:
                question = q_core + " (A) " + answer
            lfs = [slf[0] + slf[1+index] + ")" for slf in split_lfs]
            res.append({
                'id': id_ + str(index),
                'question': question,
                'answer_index': 1 if index == answer_index else 0,
                'logical_forms': lfs
            })
        return res

    @staticmethod
    def _do_flip_answers(json):
        split_q = re.split(r' *\([A-F]\) *', json['question'])
        split_lfs = [FrictionQDatasetReader._hacky_split_lf(lf) for lf in json['logical_forms']]
        answer_index = json['answer_index']
        question_new = split_q[0] + " (A) " + split_q[2] + " (B) " + split_q[1]
        lfs_new = [slf[0] + slf[2] + " " + slf[1] + ")" for slf in split_lfs]
        id_new = json['id'] + "_flip"
        json_new = {
            'id': id_new,
            'question': question_new,
            'answer_index': 1 - answer_index,
            'logical_forms': lfs_new
        }
        return [json, json_new]

    @staticmethod
    def _do_flip_worlds(question_data):
        res = deepcopy(question_data)
        res['id'] = res['id']+"_wflip"
        if "world_extractions" in res:
            res['world_extractions'] = {'world1': res['world_extractions']['world2'],
                                        'world2': res['world_extractions']['world1']}
        if "world_literals" in res:
            res['world_literals'] = {'world1': res['world_literals']['world2'],
                                     'world2': res['world_literals']['world1']}
        res['logical_forms'] = list(reversed(res['logical_forms']))
        return [question_data, res]

    entity_name_map = {"world1": "worldone","world2":"worldtwo"}

    @staticmethod
    def _stem_phrase(phrase, stemmer):
        return re.sub(r"\w+", lambda x: stemmer.stem(x.group(0)), phrase)

    @staticmethod
    def _replace_stemmed_entities(question_data, stemmer):
        question = question_data['question']
        entities = question_data['world_extractions']
        entity_pairs = []
        for key, value in entities.items():
            if not isinstance(value, list):
                entity_pairs.append((key, value))
            else:
                [entity_pairs.append((key,v)) for v in value]
        max_words = max([len(re.findall(r"\w+", string)) for _, string in entity_pairs])
        word_pos = [[match.start(0), match.end(0)] for match in re.finditer(r'\w+', question)]
        entities_stemmed = {FrictionQDatasetReader._stem_phrase(value, stemmer):
                            FrictionQDatasetReader.entity_name_map.get(key, key) for key, value in entity_pairs}

        def substitute(str):
            replacement = entities_stemmed.get(FrictionQDatasetReader._stem_phrase(str, stemmer))
            return replacement if replacement else str

        replacements = {}
        for num_words in range(1, max_words + 1):
            for i in range(len(word_pos) - num_words + 1):
                sub = question[word_pos[i][0]:word_pos[i+num_words-1][1]]
                new_sub = substitute(sub)
                if new_sub != sub:
                    replacements[re.escape(sub)] = new_sub

        if len(replacements) == 0:
            return question_data

        pattern = "|".join(sorted(replacements.keys(), key=lambda x: -len(x)))
        regex = re.compile("\\b("+pattern+")\\b")
        res = regex.sub(lambda m: replacements[re.escape(m.group(0))], question)
        question_data['question'] = res
        return question_data

    @classmethod
    def from_params(cls, params: Params) -> 'FrictionQDatasetReader':
        lazy = params.pop('lazy', False)
        filter_type2 = params.pop('filter_type2', False)
        first_lf_only = params.pop('first_lf_only', False)
        sample = params.pop('sample', -1)
        split_question = params.pop('split_question', False)
        replace_blanks = params.pop('replace_blanks', False)
        flip_answers = params.pop('flip_answers', False)
        flip_worlds = params.pop('flip_worlds', False)
        use_extracted_world_entities = params.pop('use_extracted_world_entities', False)
        extract_world_entities = params.pop('extract_world_entities', False)
        replace_world_entities = params.pop('replace_world_entities', False)
        skip_world_alignment = params.pop('skip_world_alignment', False)
        single_lf_extractor_aligned = params.pop('single_lf_extractor_aligned', False)
        gold_worlds = params.pop('gold_worlds', False)
        tagger_only = params.pop('tagger_only', False)
        entity_tag_mode = params.pop('entity_tag_mode', None)
        entity_types = params.pop('entity_types', None)
        collapse_world_tags = params.pop('collapse_world_tags', False)
        collapse_tags = params.pop('collapse_tags', None)
        if collapse_world_tags:
            collapse_tags = ['world']  # backwards compatibility
        denotation_only = params.pop('denotation_only', False)
        skip_attributes_regex = params.pop('skip_attributes_regex', None)
        lf_syntax = params.pop('lf_syntax', None)
        lexical_cues = params.pop('lexical_cues', None)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        question_token_indexers = TokenIndexer.dict_from_params(params.pop('question_token_indexers', {}))
        params.assert_empty(cls.__name__)
        return FrictionQDatasetReader(lazy=lazy,
                                      filter_type2=filter_type2,
                                      first_lf_only=first_lf_only,
                                      sample=sample,
                                      split_question=split_question,
                                      replace_blanks=replace_blanks,
                                      flip_answers=flip_answers,
                                      flip_worlds=flip_worlds,
                                      use_extracted_world_entities=use_extracted_world_entities,
                                      extract_world_entities=extract_world_entities,
                                      replace_world_entities=replace_world_entities,
                                      skip_world_alignment=skip_world_alignment,
                                      single_lf_extractor_aligned=single_lf_extractor_aligned,
                                      gold_worlds=gold_worlds,
                                      tagger_only=tagger_only,
                                      lf_syntax=lf_syntax,
                                      entity_tag_mode=entity_tag_mode,
                                      collapse_tags=collapse_tags,
                                      denotation_only=denotation_only,
                                      skip_attributes_regex=skip_attributes_regex,
                                      entity_types=entity_types,
                                      tokenizer=tokenizer,
                                      lexical_cues=lexical_cues,
                                      question_token_indexers=question_token_indexers)
