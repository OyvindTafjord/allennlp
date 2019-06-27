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
# TagSpanType = ((int, int), str)

@DatasetReader.register("bert_mc_qa")
class BertMCQAReader(DatasetReader):
    """

    Parameters
    ----------
    """

    def __init__(self,
                 pretrained_model: str,
                 instance_per_choice: bool = False,
                 max_pieces: int = 512,
                 num_choices: int = 4,
                 answer_only: bool = False,
                 syntax: str = "arc",
                 restrict_num_choices: int = None,
                 skip_id_regex: str = None,
                 ignore_context: bool = False,
                 skip_and_offset: List[int] = None,
                 annotation_tags: List[str] = None,
                 context_strip_sep: str = None,
                 context_syntax: str = "c#q#a",
                 dann_mode: bool = False,
                 sample: int = -1) -> None:
        super().__init__()
        self._token_indexers = {'tokens': SingleIdTokenIndexer()}
        lower_case = not '-cased' in pretrained_model
        self._word_splitter = WordpieceWordSplitter(pretrained_model, do_lower_case=lower_case)
        self._max_pieces = max_pieces
        self._instance_per_choice = instance_per_choice
        self._sample = sample
        self._num_choices = num_choices
        self._syntax = syntax
        self._context_syntax = context_syntax
        self._answer_only = answer_only
        self._restrict_num_choices = restrict_num_choices
        self._skip_id_regex = skip_id_regex
        self._ignore_context = ignore_context
        self._skip_and_offset = skip_and_offset
        self._context_strip_sep = context_strip_sep
        self._annotation_tags = annotation_tags
        self._dann_mode = dann_mode
        if self._annotation_tags is not None:
            self._tokenizer = WordTokenizer()
            self._num_annotation_tags = len(self._annotation_tags)
            self._annotation_tag_index = {tag:idx for idx, tag in enumerate(self._annotation_tags)}


    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        counter = self._sample + 1
        debug = 5
        offset_tracker = None
        if self._skip_and_offset is not None:
            offset_tracker = self._skip_and_offset[1]

        with open(file_path, 'r') as data_file:
            logger.info("Reading QA instances from jsonl dataset at: %s", file_path)
            for line in data_file:
                counter -= 1
                debug -= 1
                if counter == 0:
                    break
                item_json = json.loads(line.strip())
                if offset_tracker is not None:
                    if offset_tracker == 0:
                        offset_tracker = self._skip_and_offset[0] - 1
                        continue
                    else:
                        offset_tracker -= 1

                if debug > 0:
                    logger.info(item_json)

                if self._syntax == 'quarel' or self._syntax == 'quarel_preamble':
                    item_json = self._normalize_mc(item_json)
                    if debug > 0:
                        logger.info(item_json)
                elif self._syntax == 'vcr':
                    item_json = self._normalize_vcr(item_json)
                    if debug > 0:
                        logger.info(item_json)

                item_id = item_json["id"]
                if self._skip_id_regex and re.match(self._skip_id_regex, item_id):
                    continue
                context = item_json.get("para")
                if self._ignore_context:
                    context = None
                context_annotations = None
                if context is not None and self._annotation_tags is not None:
                    para_tagging = item_json.get("para_tagging")
                    context_annotations = self._get_tagged_spans(para_tagging)
                question_text = item_json["question"]["stem"]
                question_tagging = item_json.get("question_tagging")
                question_stem_annotations = None
                if question_tagging is not None and self._annotation_tags is not None:
                    question_stem_annotations = self._get_tagged_spans(question_tagging.get("stem"))

                if self._context_strip_sep is not None and context is not None:
                    split = context.split(self._context_strip_sep, 1)
                    if len(split) > 1:
                        context = split[1]

                if self._syntax == 'quarel_preamble':
                    context, question_text = question_text.split(". ", 1)

                if self._answer_only:
                    question_text = ""

                choice_label_to_id = {}
                choice_text_list = []
                choice_context_list = []
                choice_label_list = []
                choice_annotations_list = []

                any_correct = False
                choice_id_correction = 0
                choice_tagging = None
                if question_tagging is not None and self._annotation_tags is not None:
                    choice_tagging = question_tagging['choices']

                for choice_id, choice_item in enumerate(item_json["question"]["choices"]):

                    if self._restrict_num_choices and len(choice_text_list) == self._restrict_num_choices:
                        if not any_correct:
                            choice_text_list.pop(-1)
                            choice_context_list.pop(-1)
                            choice_id_correction += 1
                        else:
                            break

                    choice_label = choice_item["label"]
                    choice_label_to_id[choice_label] = choice_id - choice_id_correction

                    choice_text = choice_item["text"]
                    choice_context = choice_item.get("para")
                    if self._ignore_context:
                        choice_context = None

                    choice_annotations = []
                    if choice_tagging is not None and self._annotation_tags is not None:
                        choice_annotations = self._get_tagged_spans(choice_tagging[choice_label])

                    choice_text_list.append(choice_text)
                    choice_context_list.append(choice_context)
                    choice_label_list.append(choice_label)
                    choice_annotations_list.append(choice_annotations)

                    is_correct = 0
                    if item_json.get('answerKey') == choice_label:
                        is_correct = 1
                        if any_correct:
                            raise ValueError("More than one correct answer found for {item_json}!")
                        any_correct = True

                    if self._restrict_num_choices \
                            and len(choice_text_list) == self._restrict_num_choices \
                            and not any_correct:
                        continue

                    if self._instance_per_choice:
                        yield self.text_to_instance_per_choice(
                            str(item_id)+'-'+str(choice_label),
                            question_text,
                            choice_text,
                            is_correct,
                            context,
                            choice_context,
                            debug)

                if not any_correct and 'answerKey' in item_json:
                    raise ValueError("No correct answer found for {item_json}!")

                if not self._instance_per_choice:
                    answer_id = choice_label_to_id[item_json["answerKey"]]
                    # Pad choices with empty strings if not right number
                    if len(choice_text_list) != self._num_choices:
                        choice_text_list = (choice_text_list + self._num_choices * [''])[:self._num_choices]
                        choice_context_list = (choice_context_list + self._num_choices * [None])[:self._num_choices]
                        if answer_id >= self._num_choices:
                            logging.warning(f"Skipping question with more than {self._num_choices} answers: {item_json}")
                            continue

                    yield self.text_to_instance(
                        item_id=item_id,
                        question=question_text,
                        choice_list=choice_text_list,
                        answer_id=answer_id,
                        context=context,
                        choice_context_list=choice_context_list,
                        context_annotations=context_annotations,
                        question_stem_annotations=question_stem_annotations,
                        choice_annotations_list=choice_annotations_list,
                        debug=debug)


    def text_to_instance_per_choice(self,  # type: ignore
                         item_id: str,
                         question: str,
                         choice: str,
                         is_correct: int,
                         context: str,
                         choice_context: str,
                         debug: int = -1) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        context = choice_context or context
        qa_tokens, segment_ids = self.bert_features_from_qa(question, choice, context)

        qa_field = TextField(qa_tokens, self._token_indexers)
        fields['question'] = qa_field
        fields['segment_ids'] = SequenceLabelField(segment_ids, qa_field)
        fields['label'] = LabelField(is_correct, skip_indexing=True)

        metadata = {
            "id": item_id,
            "question_text": question,
            "choice": choice,
            "correct_answer_index": is_correct,
            # "question_tokens": [x.text for x in question_tokens],
            # "choice_tokens_list": [[x.text for x in ct] for ct in choices_tokens_list],
        }

        fields["metadata"] = MetadataField(metadata)

        if debug > 0:
            logger.info(f"qa_tokens = {qa_tokens}")
            logger.info(f"label = {is_correct}")
            logger.info(f"segment_ids = {segment_ids}")

        return Instance(fields)


    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id: str,
                         question: str,
                         choice_list: List[str],
                         answer_id: int = None,
                         context: str = None,
                         choice_context_list: List[str] = None,
                         context_annotations = None,
                         question_stem_annotations = None,
                         choice_annotations_list = None,
                         debug: int = -1) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        qa_fields = []
        segment_ids_fields = []
        qa_tokens_list = []
        annotation_tags_fields = []
        for idx, choice in enumerate(choice_list):
            choice_context = context
            choice_annotations = []
            annotation_tags_array = None
            if question_stem_annotations is not None:
                choice_annotations = choice_annotations_list[idx]
            if choice_context_list is not None and choice_context_list[idx] is not None:
                choice_context = choice_context_list[idx]
                choice_annotations = []  # Not supported in examples yet
            if question_stem_annotations is None:
                qa_tokens, segment_ids = self.bert_features_from_qa(question, choice, choice_context)
            else:
                tmp = self.bert_features_from_qa_tags(question,
                                                        choice,
                                                        choice_context,
                                                        context_annotations,
                                                        question_stem_annotations,
                                                        choice_annotations)
                qa_tokens, segment_ids, annotations_tags = tmp
                # After transpose has shape (num_annotation_tags, len(qa_tokens))
                annotation_tags_array = numpy.array(annotations_tags).transpose()
                annotation_tags_field = ArrayField(annotation_tags_array)
                annotation_tags_fields.append(annotation_tags_field)
            qa_field = TextField(qa_tokens, self._token_indexers)
            segment_ids_field = SequenceLabelField(segment_ids, qa_field)
            qa_fields.append(qa_field)
            qa_tokens_list.append(qa_tokens)
            segment_ids_fields.append(segment_ids_field)
            if debug > 0:
                logger.info(f"qa_tokens = {qa_tokens}")
                logger.info(f"segment_ids = {segment_ids}")
                if annotation_tags_array is not None:
                    logger.info(f"annotation_tags_array = {annotation_tags_array}")


        fields['question'] = ListField(qa_fields)
        fields['segment_ids'] = ListField(segment_ids_fields)
        if answer_id is not None:
            fields['label'] = LabelField(answer_id, skip_indexing=True)

        metadata = {
            "id": item_id,
            "question_text": question,
            "choice_text_list": choice_list,
            "correct_answer_index": answer_id,
            "question_tokens_list": qa_tokens_list
            # "question_tokens": [x.text for x in question_tokens],
            # "choice_tokens_list": [[x.text for x in ct] for ct in choices_tokens_list],
        }

        if len(annotation_tags_fields) > 0:
            fields['annotation_tags'] = ListField(annotation_tags_fields)
            metadata['annotation_tags'] = [x.array for x in annotation_tags_fields]

        if debug > 0:
            logger.info(f"answer_id = {answer_id}")

        if self._dann_mode:
            domain = debug % 2
            fields['domain_label'] = LabelField(domain, skip_indexing = True)
            #if domain > 0:
            #    del fields['label']
            if debug > 0:
                logger.info(f"domain = {domain}")

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    # Returns list of (offset, label) for registered tags
    def _get_tagged_spans(self, json):
        if json is None:
            return []
        tagged_spans = json.get('tagged_spans')
        if tagged_spans is None:
            return []
        res = []
        for tagged_span in tagged_spans:
            tag_label = self._normalize_tag(tagged_span['type'])
            if tag_label is not None:
                res.append((tagged_span['offset'], tag_label))
        return res

    def _normalize_tag(self, tag):
        for tag_norm in self._annotation_tags:
            if tag_norm in tag:
                return tag_norm
        return None

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

    def _normalize_mc(self, json: JsonDict) -> JsonDict:
        split = self.split_mc_question(json['question'])
        if split is None:
            raise ValueError("No question split found for {json}!")
            return None
        answer_index = json['answer_index']
        res = {"id": json['id'],
               'question': split,
               'answerKey': split['choices'][answer_index]['label']}
        return res

    def _normalize_vcr(self, json: JsonDict) -> JsonDict:
        unisex_names = ["Avery", "Riley", "Jordan", "Angel", "Parker", "Sawyer", "Peyton",
                        "Quinn", "Blake", "Hayden", "Taylor", "Alexis", "Rowan"]
        obj = json['objects']
        qa = [json['question']] + json['answer_choices']
        qa_updated = []
        for tokens in qa:
            qa_new = []
            for token in tokens:
                if isinstance(token, str):
                    qa_new.append(token)
                else:
                    entities = []
                    for ref in token:
                        entity = obj[ref]
                        if entity == 'person':
                            entity = unisex_names[ref % len(unisex_names)]
                        entities.append(entity)
                    qa_new.append(" and ".join(entities))
            qa_updated.append(" ".join(qa_new))
        answer_index = json['answer_label']
        question = qa_updated[0]
        choices = [{'text': answer, 'label': str(idx)} for idx, answer in enumerate(qa_updated[1:])]
        return {"id": json['annot_id'],
                "question": {"stem": question, "choices": choices},
                "answerKey": str(answer_index)
                }

    @staticmethod
    def split_mc_question(question, min_choices=2):
        choice_sets = [["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"],
                       ["1", "2", "3", "4", "5"],
                       ["G", "H", "J", "K"],
                       ['a', 'b', 'c', 'd', 'e']]
        patterns = [r'\(#\)', r'#\)', r'#\.']
        for pattern in patterns:
            for choice_set in choice_sets:
                regex = pattern.replace("#","(["+"".join(choice_set)+"])")
                labels = [m.group(1) for m in re.finditer(regex, question)]
                if len(labels) >= min_choices and labels == choice_set[:len(labels)]:
                    splits = [s.strip() for s in re.split(regex, question)]
                    return {"stem": splits[0],
                            "choices": [{"text": splits[i+1],
                                         "label": splits[i]} for i in range(1, len(splits)-1, 2)]}
        return None

    def bert_features_from_qa(self, question: str, answer: str, context: str = None):
        cls_token = Token("[CLS]")
        sep_token = Token("[SEP]")
        question_tokens = self._word_splitter.split_words(question)
        if context is not None:
            context_tokens = self._word_splitter.split_words(context)
            question_tokens = context_tokens + [sep_token] + question_tokens
        choice_tokens = self._word_splitter.split_words(answer)
        question_tokens, choice_tokens = self._truncate_tokens(question_tokens, choice_tokens, self._max_pieces - 3)
        tokens = [cls_token] + question_tokens + [sep_token] + choice_tokens + [sep_token]
        segment_ids = list(itertools.repeat(0, len(question_tokens) + 2)) + \
                      list(itertools.repeat(1, len(choice_tokens) + 1))
        return tokens, segment_ids

    def _get_tags(self, tokens, tag_offsets_labels):
        tags = []
        for token in tokens:
            token_idx = token.idx
            tag = [0] * self._num_annotation_tags
            for offset, label in tag_offsets_labels:
                if offset[0] <= token_idx < offset[1]:
                    idx = self._annotation_tag_index[label]
                    tag[idx] = 1
            tags.append(tag)
        return tags

    def _tokens_and_tags(self, text, tag_offsets_labels):
        word_tokens = self._tokenizer.tokenize(text)
        tags_raw = self._get_tags(word_tokens, tag_offsets_labels)
        tokens = []
        tags = []
        for wt, tag in zip(word_tokens, tags_raw):
            wp_tokens = self._word_splitter.split_words(wt.text)
            tags += [tag] * len(wp_tokens)
            tokens += wp_tokens
        return tokens, tags

    def bert_features_from_qa_tags(self,
                                   question: str,
                                   answer: str,
                                   context: str = None,
                                   context_annotations = [],
                                   question_stem_annotations = [],
                                   choice_annotations = []):
        cls_token = Token("[CLS]")
        sep_token = Token("[SEP]")
        empty_tags = [0] * self._num_annotation_tags
        question_tokens, question_tags = self._tokens_and_tags(question, question_stem_annotations)
        if context is not None:
            context_tokens, context_tags = self._tokens_and_tags(context, context_annotations)
            question_tokens = context_tokens + [sep_token] + question_tokens
            question_tags = context_tags + [empty_tags] + question_tags
        choice_tokens, choice_tags = self._tokens_and_tags(answer, choice_annotations)
        question_tokens, choice_tokens = self._truncate_tokens(question_tokens, choice_tokens, self._max_pieces - 3)
        question_tags = question_tags[-len(question_tokens):]
        choice_tags = choice_tags[:len(choice_tokens)]
        tokens = [cls_token] + question_tokens + [sep_token] + choice_tokens + [sep_token]
        tags = [empty_tags] + question_tags + [empty_tags] + choice_tags + [empty_tags]
        segment_ids = list(itertools.repeat(0, len(question_tokens) + 2)) + \
                      list(itertools.repeat(1, len(choice_tokens) + 1))
        return tokens, segment_ids, tags
