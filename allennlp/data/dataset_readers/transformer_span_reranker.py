
from typing import Dict, List, Any
import logging

import json
import numpy as np
import os
import random
import re
from overrides import overrides
import torch

from allennlp.common.file_utils import cached_path
from allennlp.common.tqdm import Tqdm
from allennlp.data.batch import Batch
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.transformer_span_prediction import TransformerSpanPredictionReader, SpanPredictionExample
from allennlp.data.fields import Field, LabelField, TextField, MultiLabelField, ListField, ArrayField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("transformer_span_reranker")
class TransformerSpanRerankerDatasetReader(DatasetReader):
    """
    """
    def __init__(self,
                 pretrained_model: str,
                 ranker_model_path: str,
                 lazy: bool = False,
                 skip_id_regex: str = None,
                 add_prefix: Dict[str, str] = None,
                 syntax: str = None,
                 model_type: str = None,
                 is_training: bool = True,
                 dataset_dir_out: str = None,
                 top_n_contexts: int = -1,
                 min_positive_frac: float = 0.1,
                 max_pieces: int = 512,
                 doc_stride: int = 100,
                 sample: int = -1
                 ) -> None:
        super().__init__(lazy)
        from allennlp.models.transformer_mc.roberta_ict import load_model_with_cache

        self._span_reader = TransformerSpanPredictionReader(
            pretrained_model=pretrained_model,
            max_pieces=max_pieces,
            doc_stride=doc_stride,
            add_prefix=add_prefix,
            is_training=is_training,
            allow_no_answer=True,
            answer_can_be_in_question=False
        )
        self._tokenizer = self._span_reader._tokenizer
        self._tokenizer_internal = self._tokenizer.tokenizer
        # Assume same token indexer in ranker model and span prediction model
        self._token_indexers = self._span_reader._token_indexers
        self._ranker_model = load_model_with_cache(ranker_model_path)
        self._ranker_model.eval()

        self._max_pieces = max_pieces
        self._sample = sample
        self._syntax = syntax
        self._top_n_contexts = top_n_contexts
        self._min_positive_frac = min_positive_frac
        self._skip_id_regex = skip_id_regex
        self._add_prefix = add_prefix or {}
        self._is_training = is_training
        self._dataset_dir_out = dataset_dir_out
        self._model_type = model_type
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
        done = False
        debug = 5

        with open(file_path, 'r') as data_file:
            logger.info("Reading instances from jsonl dataset at: %s", file_path)
            for line in data_file:
                if done:
                    break
                item_json = json.loads(line.strip())

                q_id = item_json.get("id", "NA")
                if self._skip_id_regex and re.match(self._skip_id_regex, q_id):
                    continue
                question_text = item_json['question']
                contexts = item_json.get('contexts')
                if self._top_n_contexts > 0:
                    contexts = contexts[:self._top_n_contexts]
                if debug > 0:
                    logger.info(f"id = {q_id}")
                    logger.info(f"question = {question_text}")

                if self._is_training and self._min_positive_frac > 0:
                    has_answers = [1 for x in contexts if len(x['answers']) > 0]
                    if len(has_answers) < self._min_positive_frac * len(contexts):
                        logger.info(f"{len(has_answers)}")
                        logger.info(self._min_positive_frac)
                        logger.info(f"{self._min_positive_frac * len(contexts)}")
                        logger.info(f"{[len(x['answers']) for x in contexts]}")
                        logger.info(f"Skipping {q_id}: too few positives")
                        continue

                query_raw_vector = self._get_query_raw_vector(question_text)
                question_text_prefixed = self._add_prefix.get("q", "") + question_text
                question_tokens = self.split_whitespace(question_text_prefixed)[0]
                full_answers = item_json.get('answers')

                for context in contexts:
                    counter -= 1
                    debug -= 1
                    if counter == 0:
                        done = True
                        break

                    if debug > 0:
                        logger.info(f"context = {context}")
                    context_text = context['text']
                    context_text_prefixed = self._add_prefix.get("c", "") + context_text
                    # context_headers = context['headers']
                    context_embedding = context['embedding']
                    answers = context.get('answers')
                    ir_score = context.get('score')

                    sub_id = q_id + "_" + context.get('doc_id', 'NA')
                    example = self.make_span_prediction_example(sub_id, question_text, question_tokens,
                                                            context_text_prefixed, answers)
                    additional_metadata = {
                        "question": question_text,
                        "context": context_text,
                        "answer_texts_orig": full_answers
                    }
                    yield self.text_to_instance(
                        item_id=sub_id,
                        example=example,
                        query_raw_vector=query_raw_vector,
                        context_embedding=context_embedding,
                        ir_score=ir_score,
                        additional_metadata=additional_metadata,
                        debug=debug)

    def _get_query_raw_vector(self, text):
        sentence_instance = Instance({"sentence": self.make_text_field(text)})
        batch = Batch([sentence_instance])
        batch.index_instances(self._ranker_model.vocab)
        self._ranker_model.eval()
        res = self._ranker_model.text_raw_vector(**batch.as_tensor_dict())
        return res[0].detach().tolist()

    def make_text_field(self, text: str):
        # Hackily add special tokens here since we're reusing tokenizer without them
        tokens = self._tokenizer.tokenize("<s> " + text + " </s>")
        return TextField(tokens, self._token_indexers)

    # This all can be streamlined, but using inherited old code for now
    def split_whitespace(self, text):

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    tokens.append(c)
                else:
                    tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(tokens) - 1)
        return tokens, char_to_word_offset

    def make_span_prediction_example(self, qas_id, question_text, question_tokens, context_text, answers):
        tokens, char_to_word_offset = self.split_whitespace((context_text))
        start_position = None
        end_position = None
        orig_answer_text = None
        is_impossible = False
        all_answer_texts = None
        answer_in_passage = True
        if answers is not None:
            is_impossible = len(answers) == 0
            if not is_impossible:
                all_answer_texts = [a['text'] for a in answers]
                answer = answers[0]   # Use first answer for span labeling
                orig_answer_text = answer["text"]
                answer_offset = answer["text_pos"][0]
                answer_length = len(orig_answer_text)
                start_position = char_to_word_offset[answer_offset]
                end_position = char_to_word_offset[answer_offset + answer_length - 1]

                actual_text = " ".join(tokens[start_position:(end_position + 1)])
                cleaned_answer_text = " ".join(self.split_whitespace(orig_answer_text)[0])
                if actual_text.find(cleaned_answer_text) == -1:
                    logger.warning("Could not find answer: '%s' vs. '%s'",
                                   actual_text, cleaned_answer_text)
                    if self._is_training:
                        return None
            else:
                start_position = -1
                end_position = -1
                orig_answer_text = ""

        example = SpanPredictionExample(
            qas_id=qas_id,
            doc_text=context_text,
            question_text=question_text,
            doc_tokens=tokens,
            question_tokens=question_tokens,
            answer_in_passage=answer_in_passage,
            orig_answer_text=orig_answer_text,
            all_answer_texts=all_answer_texts,
            start_position=start_position,
            end_position=end_position,
            is_impossible=is_impossible)
        return example

    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id: str,
                         example: SpanPredictionExample,
                         query_raw_vector: List[float],
                         context_embedding: List[float] = None,
                         ir_score: float = None,
                         additional_metadata: Dict[str, Any] = None,
                         debug: int = -1) -> Instance:

        fields: Dict[str, Field] = {}
        features = self._span_reader._transformer_features_from_example(example, debug)
        if features is None:
            # Shouldn't happen
            logger.info(f"NONE example = {example}")
        tokens = [Token(t, text_id=self._tokenizer_internal._convert_token_to_id(t)) for t in features.tokens]
        tokens_field = TextField(tokens, self._token_indexers)
        segment_ids_field = SequenceLabelField(features.segment_ids, tokens_field)

        fields['tokens'] = tokens_field
        fields['segment_ids'] = segment_ids_field
        fields['query_raw_vector'] = ArrayField(np.array(query_raw_vector))
        fields['block_embedding'] = ArrayField(np.array(context_embedding))

        metadata = additional_metadata or {}
        metadata.update({
            "id": item_id,
            "tokens": [x for x in features.tokens],
            "answer_texts": example.all_answer_texts,
            "answer_mask": features.p_mask,
            "is_impossible": features.is_impossible
        })

        if features.start_position is not None:
            fields['start_positions'] = LabelField(features.start_position, skip_indexing=True)
            fields['end_positions'] = LabelField(features.end_position, skip_indexing=True)
            metadata['start_positions'] = features.start_position
            metadata['end_positions'] = features.end_position
            ranker_label = 0 if features.is_impossible else 0
            fields['ranker_label'] = LabelField(ranker_label, skip_indexing=True)


        if debug > 0:
            logger.info(f"tokens = {features.tokens}")
            logger.info(f"segment_ids = {features.segment_ids}")
            logger.info(f"context = {example.doc_text}")
            logger.info(f"question = {example.question_text}")
            logger.info(f"answer_mask = {features.p_mask}")
            if features.start_position is not None and features.start_position >= 0:
                logger.info(f"start_position = {features.start_position}")
                logger.info(f"end_position = {features.end_position}")
                logger.info(f"orig_answer_text   = {example.orig_answer_text}")
                answer_text = self._span_reader._string_from_tokens(
                    features.tokens[features.start_position:(features.end_position + 1)])
                logger.info(f"answer from tokens = {answer_text}")
                logger.info(f"ranker_label = {ranker_label}")
                logger.info(f"ir_score = {ir_score}")
                query_embedding = self._ranker_model._sentence_projection(torch.Tensor([query_raw_vector]))[0]
                new_ir_score = (torch.Tensor(context_embedding) * query_embedding).sum()
                new_ir_score += self._ranker_model._classifier_bias.item()
                new_ir_score = torch.sigmoid(new_ir_score).item()
                logger.info(f"derived ir_score = {new_ir_score}")

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)
