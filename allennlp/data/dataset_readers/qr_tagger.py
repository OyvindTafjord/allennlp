from typing import Dict, List, Any
import itertools
import json
import logging
import numpy
import re

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.util import JsonDict
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, Field, TextField, LabelField
from allennlp.data.fields import ListField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import WordpieceWordSplitter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("qr_tagger")
class QRTaggerReader(DatasetReader):
    """

    """

    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tag_set: List[str] = None,
                 attention_model: bool = False,
                 max_pieces: int = None,
                 pretrained_model: str = None,
                 sample: int = -1) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        if attention_model:
            lower_case = not '-cased' in pretrained_model
            self._word_splitter = WordpieceWordSplitter(pretrained_model, do_lower_case=lower_case)

        self._tag_set = tag_set
        self._sample = sample
        self._attention_model = attention_model
        self._max_pieces = max_pieces


    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        counter = self._sample + 1
        debug = 5

        with open(file_path, 'r') as data_file:
            logger.info("Reading tagging instances from jsonl dataset at: %s", file_path)
            for line in data_file:
                counter -= 1
                debug -= 1
                if counter == 0:
                    break
                item_json = json.loads(line.strip())

                if debug > 0:
                    logger.info(item_json)

                item_id = item_json["id"]

                text = item_json["text"]
                tagged_spans = item_json["tagged_spans"]
                tag_labels = []
                tag_offsets = []
                for tagged_span in tagged_spans:
                    tag_label = self._normalize_tag(tagged_span['type'])
                    if tag_label is not None:
                        tag_labels.append(tag_label)
                        tag_offsets.append(tagged_span['offset'])
                qr_sign = item_json["qr_sign"]

                yield self.text_to_instance(
                    item_id,
                    text,
                    tag_offsets,
                    tag_labels,
                    qr_sign,
                    debug)


    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id: str,
                         text: str,
                         tag_offsets: List[List[int]] = None,
                         tag_labels: List[str] = None,
                         qr_sign: int = None,
                         debug: int = -1) -> Instance:

        if self._attention_model:
            return self._text_to_instance_attention(
                item_id,
                text,
                tag_offsets,
                tag_labels,
                qr_sign,
                debug)
        else:
            return self._text_to_instance_orig(
                item_id,
                text,
                tag_offsets,
                tag_labels,
                qr_sign,
                debug)

    def _text_to_instance_attention(self,  # type: ignore
                          item_id: str,
                          text: str,
                          tag_offsets: List[List[int]] = None,
                          tag_labels: List[str] = None,
                          qr_sign: int = None,
                          debug: int = -1) -> Instance:

        # Hack to catch separator:
        text_split = text.split("sepsep", 1)
        text_stem = text_split[0].strip()
        text_tail = text_split[1].strip() if len(text_split) > 1 else None

        text_tokens, segment_ids, attentions = \
            self.bert_features_from_text(text_stem, text_tail, tag_offsets, tag_labels)

        text_field = TextField(text_tokens, self._token_indexers)
        segment_ids_field = SequenceLabelField(segment_ids, text_field)
        attentions_field = SequenceLabelField(attentions, text_field)
        fields = {}
        #fields['question'] = ListField([text_field])
        #fields['segment_ids'] = ListField([segment_ids_field])
        fields['question'] = text_field
        fields['segment_ids'] = segment_ids_field
        fields['tags'] = attentions_field
        if qr_sign is not None and qr_sign != 0:
            label = 0
            if qr_sign > 0:
                label = 1
            fields['label'] = LabelField(label, skip_indexing=True)

        metadata = {
            "id": item_id,
            "text": text,
            "tags": attentions,
            "qr_sign": qr_sign,
            "words": [x.text for x in text_tokens]
        }

        if debug > 0:
            logger.info(f"tokens = {text_tokens}")
            logger.info(f"segment_ids = {segment_ids}")
            logger.info(f"attentions = {attentions}")
            logger.info(f"tag_labels = {tag_labels}")
            logger.info(f"qr_sign = {qr_sign}")

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    def _text_to_instance_orig(self,  # type: ignore
                     item_id: str,
                     text: str,
                     tag_offsets: List[List[int]] = None,
                     tag_labels: List[str] = None,
                     qr_sign: int = None,
                     debug: int = -1) -> Instance:

        tokenized_text = self._tokenizer.tokenize(text)

        # Silly hack for now
        for i, t in enumerate(tokenized_text):
            if t.text == "sepsep":
                tokenized_text[i] = Token(text="[SEP]", idx=t.idx)

        tags = None
        if tag_offsets is not None:
            tags = self._get_bio_tags(tokenized_text, tag_offsets, tag_labels)

        cls_token = Token("[CLS]")
        sep_token = Token("[SEP]")
        if "qr_sign" in self._tag_set:
            tokenized_text = [cls_token] + tokenized_text + [sep_token]
            if tags is not None:
                sign_tag = "B-QRPlus" if qr_sign > 0 else "B-QRMinus"
                tags = [sign_tag] + tags + ["O"]

        text_field = TextField(tokenized_text, self._token_indexers)

        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        fields['tokens'] = text_field
        if tags is not None:
            fields['tags'] = SequenceLabelField(tags, text_field)

        metadata = {
            "id": item_id,
            "text": text,
            "tags": tags,
            "qr_sign": qr_sign,
            "words": [x.text for x in tokenized_text]
        }

        if debug > 0:
            logger.info(f"tokens = {tokenized_text}")
            logger.info(f"tags = {tags}")
            logger.info(f"tag_labels = {tag_labels}")

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    def _normalize_tag(self, tag):
        for tag_norm in self._tag_set:
            if tag_norm in tag:
                return tag_norm
        return None

    def _get_bio_tags(self, tokens, tag_offsets, tag_labels):
        current_tag = "O"
        tags = []

        for token in tokens:
            token_idx = token.idx
            tag = None
            for offset, label in zip(tag_offsets, tag_labels):
                if offset[0] <= token_idx < offset[1]:
                    tag = label
                    break
            if tag:
                if current_tag in ["B-"+tag, "I-"+tag]:
                    tag = "I-"+tag
                else:
                    tag = "B-"+tag
            else:
                tag = "O"
            current_tag = tag
            tags.append(tag)
        return tags

    @staticmethod
    def _truncate_tokens(tokens_a, tokens_b, max_length):
        """
        Truncate a from the start and b from the end until total is less than max_length.
        At each step, truncate the longest one
        """
        while len(tokens_a) + len(tokens_b) > max_length:
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop(0)
            else:
                tokens_b.pop()
        return tokens_a, tokens_b

    def bert_features_from_text(self, stem: str, tail: str, tag_offsets, tag_labels):
        cls_token = Token("[CLS]")
        sep_token = Token("[SEP]")
        stem_word_tokens = self._tokenizer.tokenize(stem)
        tags_raw = self._get_bio_tags(stem_word_tokens, tag_offsets, tag_labels)
        stem_tokens = []
        attention_filter = []
        for wt, tag in zip(stem_word_tokens, tags_raw):
            wp_tokens = self._word_splitter.split_words(wt.text)
            attention = 0 if tag == "O" else 1
            attention_filter += [attention] * len(wp_tokens)
            stem_tokens += wp_tokens
        if tail is not None:
            tail_tokens = self._word_splitter.split_words(tail) + [sep_token]
        else:
            tail_tokens = []
        stem_tokens, tail_tokens = self._truncate_tokens(stem_tokens, tail_tokens, self._max_pieces - 3)
        tokens = [cls_token] + stem_tokens + [sep_token] + tail_tokens
        segment_ids = [0] * (len(stem_tokens) + 2) + [1] * (len(tail_tokens))
        attention_filter = [0] + attention_filter + [0] + [0] * len(tail_tokens)
        return tokens, segment_ids, attention_filter