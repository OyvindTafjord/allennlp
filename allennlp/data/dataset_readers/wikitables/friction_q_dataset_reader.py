"""
Reader for Friction questions
"""

from typing import Any, Dict, List, Union
import json
import logging

from overrides import overrides

import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset import Dataset
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import TextField, KnowledgeGraphField, LabelField, ListField, MetadataField
from allennlp.data.semparse.knowledge_graphs import TableKnowledgeGraph
from allennlp.data.semparse.worlds import FrictionWorld
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.seq2seq import START_SYMBOL, END_SYMBOL

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("friction_q")
class FrictionQDatasetReader(DatasetReader):
    """

    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 question_token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._tokenizer = tokenizer or WordTokenizer()
        self._question_token_indexers = question_token_indexers or {"tokens": SingleIdTokenIndexer()}
        # Fake table so we can use WikiTable parser model
        self._table_knowledge_graph = TableKnowledgeGraph.read_from_json(
            {"columns":["foo"], "cells": [["foo"]]})
        self._world = FrictionWorld()
        self._table_token_indexers = self._question_token_indexers

    @overrides
    def read(self, file_path):
        instances = []
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file: %s", file_path)
            for line in tqdm.tqdm(data_file):
                line = line.strip("\n")
                if not line:
                    continue
                question_data = json.loads(line)
                question = question_data['question']
                question_id = question_data['id']
                logical_forms = question_data['logical_forms']
                answer_index = question_data['answer_index']
                additional_metadata = {'id': question_id, 'answer_index': answer_index}
                instances.append(self.text_to_instance(question, logical_forms, additional_metadata))
        if not instances:
            raise ConfigurationError("No instances read!")
        return Dataset(instances)

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
        table_field = KnowledgeGraphField(self._table_knowledge_graph, self._table_token_indexers)
        fields = {'question': question_field, 'table': table_field}
        if logical_forms:
            expressions = [self._world.parse_logical_form(form) for form in logical_forms]
            action_sequences = [self._world.get_action_sequence(expression) for expression in expressions]
            action_sequences_field = ListField([self._make_action_sequence_field(sequence)
                                                for sequence in action_sequences])
            fields['target_action_sequences'] = action_sequences_field
        # fields['metadata'] = MetadataField(additional_metadata or {})
        return Instance(fields)

    @staticmethod
    def _make_action_sequence_field(action_sequence: List[str]) -> ListField:
        action_sequence.insert(0, START_SYMBOL)
        action_sequence.append(END_SYMBOL)
        return ListField([LabelField(action, label_namespace='actions') for action in action_sequence])

    @classmethod
    def from_params(cls, params: Params) -> 'FrictionQDatasetReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        question_token_indexers = TokenIndexer.dict_from_params(params.pop('question_token_indexers', {}))
        params.assert_empty(cls.__name__)
        return FrictionQDatasetReader(tokenizer=tokenizer,
                                      question_token_indexers=question_token_indexers)
