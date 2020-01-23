from typing import cast, Tuple

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.data.dataset_readers import TransformerMCQAReader
from allennlp.predictors.predictor import Predictor

@Predictor.register('rule-reasoning')
class RuleReasoningPredictor(Predictor):
    """
    Wrapper for the bert_mc_qa model.
    """
    def _my_json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        """

        # Make a cast here to satisfy mypy
        dataset_reader = cast(TransformerMCQAReader, self._dataset_reader)

        question_raw = json_dict['question']
        question_mc = None
        if isinstance(question_raw, str):
            question_mc = question_raw
            question_data = dataset_reader.split_mc_question(question_raw)
            if question_data is None:
                question_data = {"stem": "Which is true?",
                                 "choices": [{"label": "A", "text": question_raw}]}
        else:
            question_data = question_raw
        question_text = question_data["stem"]
        choice_text_list = [choice['text'] for choice in question_data['choices']]
        choice_labels = [choice['label'] for choice in question_data['choices']]
        # support both "para" and "passage" for input context
        context = json_dict.get("para", json_dict.get("passage"))
        choice_context_list = [choice.get('para') for choice in question_data['choices']]
        if context is None and "<p>" in question_text:
            context, question_text = question_text.split("<p>", 1)
            context = context.strip()
            question_text = question_text.strip()

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

        answers = outputs['answer_index']
        scores = outputs['label_probs']

        answer_strings = []
        for label, answer, score in zip(return_dict['choice_labels'], answers, scores):
            answer_strings.append(f"({label}) {answer} [{score:.4f}]")
        outputs['answer_string'] = "  ".join(answer_strings)
        outputs['answer_strings'] = answer_strings

        return_dict.update(outputs)

        return sanitize(return_dict)
