from typing import Dict, List, Any
import itertools
import json
import logging
import numpy
import os
import re

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import JsonDict
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.document_retriever import combine_sentences, list_sentences, DocumentRetriever
from allennlp.data.fields import ArrayField, Field, TextField, LabelField
from allennlp.data.fields import ListField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
# TagSpanType = ((int, int), str)

@DatasetReader.register("transformer_classification")
class TransformerClassificationReader(DatasetReader):
    """

    Parameters
    ----------
    """

    def __init__(self,
                 pretrained_model: str,
                 max_pieces: int = 512,
                 syntax: str = "arc-ir",
                 skip_id_regex: str = None,
                 context_strip_sep: str = None,
                 answer_only: bool = False,
                 ignore_answer: bool = False,
                 context_syntax: str = "c#q#_a!",
                 add_prefix: Dict[str, str] = None,
                 document_retriever: DocumentRetriever = None,
                 override_context: bool = False,
                 context_format: Dict[str, Any] = None,
                 dataset_dir_out: str = None,
                 model_type: str = None,
                 do_lowercase: bool = None,
                 sample: int = -1) -> None:
        super().__init__()
        if do_lowercase is None:
            do_lowercase = '-uncased' in pretrained_model

        self._tokenizer = PretrainedTransformerTokenizer(pretrained_model, add_special_tokens=False)
        self._tokenizer_internal = self._tokenizer.tokenizer
        token_indexer = PretrainedTransformerIndexer(pretrained_model)
        self._token_indexers = {'tokens': token_indexer}

        self._max_pieces = max_pieces
        self._sample = sample
        self._syntax = syntax
        self._context_syntax = context_syntax
        self._answer_only = answer_only
        self._ignore_answer = ignore_answer
        self._skip_id_regex = skip_id_regex
        self._override_context = override_context
        self._context_strip_sep = context_strip_sep
        self.document_retriever = document_retriever
        self._context_format = context_format
        self._dataset_dir_out = dataset_dir_out
        self._model_type = model_type
        self._add_prefix = add_prefix or {}
        if model_type is None:
            for model in ["roberta", "bert", "openai-gpt", "gpt2", "transfo-xl", "xlnet", "xlm"]:
                if model in pretrained_model:
                    self._model_type = model
                    break

    @overrides
    def _read(self, file_path: str):
        self._dataset_cache = None
        if self._dataset_dir_out is not None:
            self._dataset_cache = []
        instances = self._read_internal(file_path)
        if self.document_retriever is not None:
            if not isinstance(instances, list):
                instances = [instance for instance in Tqdm.tqdm(instances)]
            cfo = self.document_retriever._cache_file_out
            if cfo is not None:
                logger.info(f"Saving document retriever cache to {cfo}.")
                self.document_retriever.save_cache_file()
        if self._dataset_cache is not None:
            if not isinstance(instances, list):
                instances = [instance for instance in Tqdm.tqdm(instances)]
            if not os.path.exists(self._dataset_dir_out):
                os.mkdir(self._dataset_dir_out)
            output_file = os.path.join(self._dataset_dir_out, os.path.basename(file_path))
            logger.info(f"Saving contextualized dataset to {output_file}.")
            with open(output_file, 'w') as file:
                for d in self._dataset_cache:
                    file.write(json.dumps(d))
                    file.write("\n")
        return instances

    def _read_internal(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        counter = self._sample + 1
        debug = 5

        with open(file_path, 'r') as data_file:
            logger.info("Reading instances from jsonl dataset at: %s", file_path)
            for line in data_file:
                item_json = json.loads(line.strip())

                item_id = item_json.get("id", "NA")
                if self._skip_id_regex and re.match(self._skip_id_regex, item_id):
                    continue

                counter -= 1
                debug -= 1
                if counter == 0:
                    break

                if debug > 0:
                    logger.info(item_json)

                context = item_json.get("context")
                if self._syntax == "arc-ir":
                    question_text = item_json["question_text"]
                    choice = item_json.get("choice")
                    label = item_json.get("label")
                elif self._syntax == "lmbias":
                    question_text = item_json["text"]
                    choice = None
                    label = item_json.get("model")
                else:
                    raise ValueError(f"Unknown syntax {self._syntax}")

                if self._ignore_answer:
                    choice = None

                if (context is None or self._override_context) and self._context_format is not None:
                    raise NotImplementedError
                if self._context_strip_sep is not None and context is not None:
                    split = context.split(self._context_strip_sep, 1)
                    if len(split) > 1:
                        context = split[1]

                if self._answer_only:
                    question_text = ""

                if self._dataset_cache is not None:
                    self._dataset_cache.append(item_json)

                additional_metadata = {}
                for key in ["orig_logit", "is_correct"]:
                    if key in item_json:
                        additional_metadata[key] = item_json[key]


                yield self.text_to_instance(
                    item_id=item_id,
                    question=question_text,
                    choice=choice,
                    label=label,
                    context=context,
                    additional_metadata=additional_metadata,
                    debug=debug)

    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id: str,
                         question: str,
                         choice: str = None,
                         label: int = None,
                         context: str = None,
                         additional_metadata: Dict[str, Any] = {},
                         debug: int = -1) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        qa_tokens, segment_ids = self.transformer_features_from_qa(question, choice, context)
        qa_field = TextField(qa_tokens, self._token_indexers)
        segment_ids_field = SequenceLabelField(segment_ids, qa_field)
        fields['tokens'] = qa_field
        fields['segment_ids'] = segment_ids_field

        metadata = {
            "id": item_id,
            "question_text": question,
            "choice": choice,
            "tokens": [x.text for x in qa_tokens],
            "context": context
        }
        metadata.update(additional_metadata)

        if label is not None:
            # We'll assume integer labels don't need indexing
            fields['label'] = LabelField(label, skip_indexing=isinstance(label, int))
            metadata['label'] = label

        if debug > 0:
            logger.info(f"qa_tokens = {qa_tokens}")
            logger.info(f"segment_ids = {segment_ids}")
            logger.info(f"context = {context}")
            logger.info(f"choice = {choice}")
            logger.info(f"label = {label}")

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    @staticmethod
    def _truncate_tokens(context_tokens, question_tokens, choice_tokens, max_length):
        """
        Truncate context_tokens first, from the left, then question_tokens and choice_tokens
        """
        max_context_len = max_length - len(question_tokens) - len(choice_tokens)
        if max_context_len > 0:
            if len(context_tokens) > max_context_len:
                context_tokens = context_tokens[-max_context_len:]
        else:
            context_tokens = []
            while len(question_tokens) + len(choice_tokens) > max_length:
                if len(question_tokens) > len(choice_tokens):
                    question_tokens.pop(0)
                else:
                    choice_tokens.pop()
        return context_tokens, question_tokens, choice_tokens


    def transformer_features_from_qa(self, question: str, answer: str, context: str = None):
        cls_token = self._tokenizer.tokenize(self._tokenizer_internal.cls_token)[0]
        sep_token =  self._tokenizer.tokenize(self._tokenizer_internal.sep_token)[0]
        sep_token_extra = bool(self._model_type in ['roberta'])
        cls_token_at_end = bool(self._model_type in ['xlnet'])
        cls_token_segment_id = 2 if self._model_type in ['xlnet'] else 0
        question = self._add_prefix.get("q", "") + question
        if answer is not None:
            answer = self._add_prefix.get("a",  "") + answer
        question_tokens = self._tokenizer.tokenize(question)
        if context is not None:
            context = self._add_prefix.get("c", "") + context
            context_tokens = self._tokenizer.tokenize(context)
        else:
            context_tokens = []

        seps = self._context_syntax.count("#")
        sep_mult = 2 if sep_token_extra else 1
        max_tokens = self._max_pieces - seps * sep_mult - 1

        if answer is not None:
            choice_tokens = self._tokenizer.tokenize(answer)
        else:
            choice_tokens = []

        context_tokens, question_tokens, choice_tokens = self._truncate_tokens(context_tokens,
                                                                               question_tokens,
                                                                               choice_tokens,
                                                                               max_tokens)
        tokens = []
        segment_ids = []
        current_segment = 0
        token_dict = {"q": question_tokens, "c": context_tokens, "a": choice_tokens}
        for c in self._context_syntax:
            if c in "qca":
                new_tokens = token_dict[c]
                tokens += new_tokens
                segment_ids += len(new_tokens) * [current_segment]
            elif c == "#":
                tokens += sep_mult * [sep_token]
                segment_ids += sep_mult * [current_segment]
            elif c == "!":
                tokens += [sep_token]
                segment_ids += [current_segment]
            elif c == "_":
                current_segment += 1
            else:
                raise ValueError(f"Unknown context_syntax character {c} in {self._context_syntax}")

        if cls_token_at_end:
            tokens += [cls_token]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        return tokens, segment_ids
