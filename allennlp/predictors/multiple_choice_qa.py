from typing import cast, Tuple

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.data.dataset_readers import BertMCQAReader
from allennlp.predictors.predictor import Predictor

# Some util functions which can be moved out

ATTENTION_CUTOFF = 0.5
MAX_NUM_SPANS = 6


def merge_wp_tokens(tokens):
    res = ""
    for token in tokens:
        if token.startswith("##"):
            res += token[2:]
        elif res == "":
            res = token
        else:
            res += " " + token
    return res


tag_explanation_lookup = {"dir": "Qualitative direction", "prop": "Qualitative property"}


def get_qr_explanation(res, tags, cutoff):
    max_num_spans = MAX_NUM_SPANS
    answer_index = res['answer_index']
    context = res.get('context')
    if context is None:
        context = res.get('choice_context_list')
        if context is not None:
            context = context[answer_index]
        if context is None:
            context = "NA"
    explanations = []
    explanations.append({"header": "Retrieved knowledge", "content": context})
    qa_tokens = res['question_tokens_list'][answer_index]
    attentions = res['annotation_attentions'][answer_index]
    for idx, tag in enumerate(tags):
        top_spans = get_top_spans(qa_tokens, attentions[idx], cutoff)
        seen = set()
        top_spans_content = []
        for span in top_spans:
            text = merge_wp_tokens(span['tokens'])
            if text in seen:
                continue
            seen.add(text)
            top_spans_content.append(f'{text} ({span["score"]:.2f})')
            if len(top_spans_content) >= max_num_spans:
                break
        if len(top_spans) > 0:
            pretty_tag = tag_explanation_lookup.get(tag, tag)
            explanations.append({"header": f"Identified spans for {pretty_tag}:", "content": top_spans_content})
    return explanations


def get_top_spans(tokens, attentions, cutoff):
    spans = []
    current_score = 0
    for token, attention in zip(tokens, attentions):
        if attention > cutoff:
            if current_score == 0:
                current_score = attention
                spans.append({"tokens": [token], "score": current_score})
            else:
                current_score += attention
                spans[-1]['tokens'].append(token)
                spans[-1]['score'] = current_score
        else:
            current_score = 0
    spans.sort(key=lambda x: -x['score'])
    return spans


@Predictor.register('multiple-choice-qa')
class MultipleChoiceQAPredictor(Predictor):
    """
    Wrapper for the bert_mc_qa model.
    """
    def _my_json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        """

        # Make a cast here to satisfy mypy
        dataset_reader = cast(BertMCQAReader, self._dataset_reader)

        question_raw = json_dict['question']
        question_mc = None
        if isinstance(question_raw, str):
            question_mc = question_raw
            question_data = dataset_reader.split_mc_question(question_raw)
        else:
            question_data = question_raw
        question_text = question_data["stem"]
        choice_text_list = [choice['text'] for choice in question_data['choices']]
        choice_labels = [choice['label'] for choice in question_data['choices']]
        context = json_dict.get("para")
        choice_context_list = [choice.get('para') for choice in question_data['choices']]
        no_context = context is None and choice_context_list[0] is None
        if no_context and dataset_reader._context_format is not None:
            context = dataset_reader._get_q_context(question_text, choice_text_list)
            choice_context_list = [dataset_reader._get_qa_context(question_text, choice) for choice in choice_text_list]

        instance = dataset_reader.text_to_instance(
            "NA",
            question_text,
            choice_text_list,
            context=context,
            choice_context_list=choice_context_list
        )

        if question_mc is None:
            question_mc = " ".join([question_text, *[f"(x) y" for x,y in zip(choice_labels, choice_text_list)]])

        extra_info = {'id': question_data.get('id', "NA"),
                      'question': question_mc,
                      'choice_labels': choice_labels,
                      'context': context,
                      'annotation_tags': dataset_reader._annotation_tags,
                      'choice_context_list': choice_context_list}

        return instance, extra_info

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        instance, _ = self._my_json_to_instance(json_dict)
        return instance

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance, return_dict = self._my_json_to_instance(inputs)
        outputs = self._model.forward_on_instance(instance)

        answer_index = outputs['answer_index']
        answer = return_dict['choice_labels'][answer_index]
        score = max(outputs['label_probs'])

        outputs['answer'] = answer
        outputs['score'] = score
        outputs['question_tokens_list'] = sanitize(instance.fields['metadata']['question_tokens_list'])

        # Aristo system format predictions
        labels_scores = zip(return_dict['choice_labels'], outputs['label_probs'])
        choices = [{"label": x, "score": y} for x,y in labels_scores]
        outputs['prediction'] = {"choices": choices}

        if "annotation_attentions" in outputs:
            qr_explanation = get_qr_explanation(outputs, return_dict['annotation_tags'], ATTENTION_CUTOFF)
            outputs['qr_explanation'] = qr_explanation

        return_dict.update(outputs)

        return sanitize(return_dict)
