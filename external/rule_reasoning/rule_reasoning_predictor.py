from typing import cast, Dict, List, Tuple
import re

from copy import deepcopy
import numpy
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.data.fields import LabelField
from allennlp.predictors.predictor import Predictor
from external.rule_reasoning.rule_reasoning_reader import RuleReasoningReader

@Predictor.register('rule-reasoning-predictor')
class RuleReasoningPredictor(Predictor):
    """
    Wrapper for the roberta_classifier model
    """
    def _my_json_to_instances(self, json_dict: JsonDict) -> List[Instance]:
        """
        """

        # Make a cast here to satisfy mypy
        dataset_reader = cast(RuleReasoningReader, self._dataset_reader)

        question_raw = json_dict['question']
        question_list = [s.strip() for s in re.split("(?<=\\.)|\n", question_raw) if s.strip() != ""]
        context = json_dict.get("para", json_dict.get("passage"))
        if context is not None:
            #remove newlines and add periods if no other punctuation at ends of lines
            lines = context.split("\n")
            res = []
            for s in lines:
                s1 = s.strip()
                if len(s1) > 0 and s1[-1].isalnum():
                    s1 += "."
                res.append(s1)
            context = " ".join(res)

        instances = []
        for q in question_list:
            instances.append(dataset_reader.text_to_instance(
                item_id="NA",
                question_text=q,
                context=context))
        meta = {'question_list': question_list}

        return instances, meta

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        instance = self._my_json_to_instance(json_dict)
        return instance[0]

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instances, return_dict = self._my_json_to_instances(inputs)
        outputs = self._model.forward_on_instances(instances)

        answer_strings = []
        scores = []
        labels = []
        question_list = return_dict['question_list']
        for i, output in enumerate(outputs):
            answer_index = output['label_predicted']
            score = output['label_probs'][answer_index]
            question = question_list[i]
            answer = answer_index == 1
            labels.append(str(answer))
            answer_strings.append(f"{question}: {answer} [{score:.4f}]")
            scores.append(score)

        return_dict['answer_strings'] = answer_strings
        return_dict['labels'] = labels
        return_dict['scores'] = scores
        return_dict['question_list'] = question_list

        return sanitize(return_dict)

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        new_instance = deepcopy(instance)
        label = numpy.argmax(outputs["label_probs"])
        new_instance.add_field("label", LabelField(int(label), skip_indexing=True))
        return [new_instance]
