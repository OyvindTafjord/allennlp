"""
Reader for Friction questions
"""

from typing import Any, Dict, List, Union
import json
import logging
import re

from overrides import overrides

import tqdm

from allennlp.common import Params
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_stemmer import PorterStemmer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import Field, TextField, KnowledgeGraphField, LabelField
from allennlp.data.fields import IndexField, ListField, MetadataField, ProductionRuleField
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.seq2seq import START_SYMBOL, END_SYMBOL
from allennlp.semparse.contexts import TableQuestionKnowledgeGraph
from allennlp.semparse.contexts.knowledge_graph import KnowledgeGraph
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
    use_extracted_world_entities : ``bool`` (optional, default=False)
        Use preprocessed world entity strings as part of the LFs
    lf_syntax: ``str``
        Which LF formalism to use, see friction types for details

    """
    def __init__(self,
                 lazy: bool = False,
                 filter_type2: bool = False,
                 first_lf_only: bool = False,
                 split_question: bool = False,
                 replace_blanks: bool = False,
                 flip_answers: bool = False,
                 lf_syntax: str = None,
                 use_extracted_world_entities: bool = False,
                 replace_world_entities: bool = False,
                 tokenizer: Tokenizer = None,
                 question_token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._question_token_indexers = question_token_indexers or {"tokens": SingleIdTokenIndexer()}
        # Fake table so we can use WikiTable parser model
        self._table_knowledge_graph = TableQuestionKnowledgeGraph.read_from_json(
            {"columns":["foo"], "cells": [["foo"]], "question":[]})
        self._world = FrictionWorld(self._table_knowledge_graph, lf_syntax)
        self._table_token_indexers = self._question_token_indexers
        self._filter_type2 = filter_type2
        self._first_lf_only = first_lf_only
        self._split_question = split_question
        self._replace_blanks = replace_blanks
        self._flip_answers = flip_answers
        self._use_extracted_world_entities = use_extracted_world_entities
        self._replace_world_entities = replace_world_entities
        self._lf_syntax = lf_syntax

        if self._replace_world_entities:
            self._stemmer = PorterStemmer().stemmer


    @overrides
    def _read(self, file_path):
        debug = 5
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file: %s", file_path)
            for line in tqdm.tqdm(data_file):
                line = line.strip("\n")
                if not line:
                    continue
                question_datas = json.loads(line)
                # Skip training instances without world entities if needing them
                if (self._use_extracted_world_entities or self._replace_world_entities) \
                        and "world_extractions" not in question_datas:
                    continue
                if (self._replace_world_entities):
                    new_q = self._replace_stemmed_entities(question_datas['question'],
                                                           question_datas['world_extractions'],
                                                           self._stemmer)
                    question_datas['question'] = new_q
                if self._split_question:
                    question_datas = self._split_instance(question_datas, self._replace_blanks)
                elif self._flip_answers:
                    question_datas = self._do_flip_answers(question_datas)
                else:
                    question_datas = [question_datas]
                debug -= 1
                if debug > 0:
                    logger.info(question_datas)
                for question_data in question_datas:
                    question = question_data['question']
                    question = self._fix_question(question)
                    question_id = question_data['id']
                    logical_forms = question_data['logical_forms']
                    if (self._first_lf_only or self._use_extracted_world_entities
                            or self._replace_world_entities) and len(logical_forms) > 1:
                        logical_forms = [logical_forms[0]]
                    # Hacky filter to ignore "type2" questions
                    if self._filter_type2 and len(logical_forms) > 0 and "(and " in logical_forms[0]:
                        continue
                    answer_index = question_data['answer_index']
                    world_extractions = question_data.get('world_extractions')
                    additional_metadata = {'id': question_id,
                                           'question': question,
                                           'answer_index': answer_index,
                                           'logical_forms': logical_forms}
                    yield self.text_to_instance(question, logical_forms,
                                                additional_metadata, world_extractions)


    @overrides
    def text_to_instance(self,  # type: ignore
                         question: str,
                         logical_forms: List[str] = None,
                         additional_metadata: Dict[str, Any] = None,
                         world_extractions: Dict[str, str] = None,
                         tokenized_question: List[Token] = None) -> Instance:
        """

        """
        # pylint: disable=arguments-differ
        tokenized_question = tokenized_question or self._tokenizer.tokenize(question.lower())
        additional_metadata['question_tokens'] = [token.text for token in tokenized_question]
        question_field = TextField(tokenized_question, self._question_token_indexers)
        if self._use_extracted_world_entities:
            neighbors = {key: [] for key in world_extractions.keys()}
            knowledge_graph = KnowledgeGraph(entities=set(world_extractions.keys()),
                                             neighbors=neighbors,
                                             entity_text=world_extractions)
            world = FrictionWorld(knowledge_graph, self._lf_syntax)
            additional_metadata['world_extractions'] = world_extractions
        else:
            knowledge_graph = self._table_knowledge_graph
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
                    logger.debug(f'Missing production rule: {error.args}, skipping logical form')
                    logger.debug(f'Question was: {question}')
                    logger.debug(f'Logical form was: {logical_form}')
                    continue
            fields['target_action_sequences'] = ListField(action_sequence_fields)
        fields['metadata'] = MetadataField(additional_metadata or {})
        return Instance(fields)

    @staticmethod
    def _fix_question(question: str) -> str:
        """
        Replace answer dividers (A), (B) etc with a unique token answeroptionA, answeroptionB, ...
        Replace '_____' with 'blankblank'
        """
        res = re.sub(r'\(([A-G])\)', r"answeroption\1", question)
        res = re.sub(r" *_{3,} *", " blankblank ", res)
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

    entity_name_map = {"world1": "worldone","world2":"worldtwo"}

    @staticmethod
    def _stem_phrase(phrase, stemmer):
        return re.sub(r"\w+", lambda x: stemmer.stem(x.group(0)), phrase)

    @staticmethod
    def _replace_stemmed_entities(question, entities, stemmer):
        max_words = max([len(re.findall(r"\w+", string)) for string in entities.values()])
        word_pos = [[match.start(0), match.end(0)] for match in re.finditer(r'\w+', question)]
        entities_stemmed = {FrictionQDatasetReader._stem_phrase(value, stemmer):
                            FrictionQDatasetReader.entity_name_map.get(key, key) for key, value in entities.items()}

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
            return question

        pattern = "|".join(sorted(replacements.keys(), key=lambda x: -len(x)))
        regex = re.compile("\\b("+pattern+")\\b")
        res = regex.sub(lambda m: replacements[re.escape(m.group(0))], question)
        return res

    @classmethod
    def from_params(cls, params: Params) -> 'FrictionQDatasetReader':
        lazy = params.pop('lazy', False)
        filter_type2 = params.pop('filter_type2', False)
        first_lf_only = params.pop('first_lf_only', False)
        split_question = params.pop('split_question', False)
        replace_blanks = params.pop('replace_blanks', False)
        flip_answers = params.pop('flip_answers', False)
        use_extracted_world_entities = params.pop('use_extracted_world_entities', False)
        replace_world_entities = params.pop('replace_world_entities', False)
        lf_syntax = params.pop('lf_syntax', None)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        question_token_indexers = TokenIndexer.dict_from_params(params.pop('question_token_indexers', {}))
        params.assert_empty(cls.__name__)
        return FrictionQDatasetReader(lazy=lazy,
                                      filter_type2=filter_type2,
                                      first_lf_only=first_lf_only,
                                      split_question=split_question,
                                      replace_blanks=replace_blanks,
                                      flip_answers=flip_answers,
                                      use_extracted_world_entities=use_extracted_world_entities,
                                      replace_world_entities=replace_world_entities,
                                      lf_syntax=lf_syntax,
                                      tokenizer=tokenizer,
                                      question_token_indexers=question_token_indexers)
