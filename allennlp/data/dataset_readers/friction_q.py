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
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import Field, TextField, KnowledgeGraphField, LabelField
from allennlp.data.fields import IndexField, ListField, MetadataField, ProductionRuleField
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.seq2seq import START_SYMBOL, END_SYMBOL
from allennlp.semparse.contexts import TableQuestionKnowledgeGraph
from allennlp.semparse.worlds import FrictionWorld


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("friction_q")
class FrictionQDatasetReader(DatasetReader):
    """

    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 question_token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._question_token_indexers = question_token_indexers or {"tokens": SingleIdTokenIndexer()}
        # Fake table so we can use WikiTable parser model
        self._table_knowledge_graph = TableQuestionKnowledgeGraph.read_from_json(
            {"columns":["foo"], "cells": [["foo"]], "question":[]})
        self._world = FrictionWorld(self._table_knowledge_graph)
        self._table_token_indexers = self._question_token_indexers

    @overrides
    def _read(self, file_path):
        instances = []
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file: %s", file_path)
            for line in tqdm.tqdm(data_file):
                line = line.strip("\n")
                if not line:
                    continue
                question_data = json.loads(line)
                question = question_data['question']
                question = self._fix_question(question)
                question_id = question_data['id']
                logical_forms = question_data['logical_forms']
                answer_index = question_data['answer_index']
                additional_metadata = {'id': question_id,
                                       'answer_index': answer_index,
                                       'logical_forms': logical_forms}
                yield self.text_to_instance(question, logical_forms, additional_metadata)


    @overrides
    def text_to_instance(self,  # type: ignore
                         question: str,
                         logical_forms: List[str] = None,
                         additional_metadata: Dict[str, Any] = None) -> Instance:
        """

        """
        # pylint: disable=arguments-differ
        tokenized_question = self._tokenizer.tokenize(question)
        question_field = TextField(tokenized_question, self._question_token_indexers)
        table_field = KnowledgeGraphField(self._table_knowledge_graph,
                                          tokenized_question,
                                          self._table_token_indexers,
                                          tokenizer=self._tokenizer)
        world = self._world
        world_field = MetadataField(world)

        production_rule_fields: List[Field] = []
        for production_rule in self._world.all_possible_actions():
            _, rule_right_side = production_rule.split(' -> ')
            is_global_rule = not self._world.is_table_entity(rule_right_side)
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
                expression = self._world.parse_logical_form(logical_form)
                action_sequence = self._world.get_action_sequence(expression)
                try:
                    index_fields: List[Field] = []
                    for production_rule in action_sequence:
                        index_fields.append(IndexField(action_map[production_rule], action_field))
                    action_sequence_fields.append(ListField(index_fields))
                except KeyError as error:
                    logger.debug(f'Missing production rule: {error.args}, skipping logical form')
                    logger.debug(f'Question was: {question}')
                    logger.debug(f'Table info was: {table_info}')
                    logger.debug(f'Logical form was: {logical_form}')
                    continue
            fields['target_action_sequences'] = ListField(action_sequence_fields)
        fields['metadata'] = MetadataField(additional_metadata or {})
        return Instance(fields)

    @staticmethod
    def _fix_question(question: str) -> str:
        """
        Replace answer dividers (A), (B) etc with a unique token answeroptionA, answeroptionB, ...
        """
        return re.sub(r'\(([A-G])\)', r"answeroption\1", question)


    @staticmethod
    def _make_action_sequence_field(action_sequence: List[str]) -> ListField:
        action_sequence.insert(0, START_SYMBOL)
        action_sequence.append(END_SYMBOL)
        return ListField([LabelField(action, label_namespace='actions') for action in action_sequence])

    @classmethod
    def from_params(cls, params: Params) -> 'FrictionQDatasetReader':
        lazy = params.pop('lazy', False)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        question_token_indexers = TokenIndexer.dict_from_params(params.pop('question_token_indexers', {}))
        params.assert_empty(cls.__name__)
        return FrictionQDatasetReader(lazy=lazy,
                                      tokenizer=tokenizer,
                                      question_token_indexers=question_token_indexers)
