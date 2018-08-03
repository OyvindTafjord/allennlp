import os
from subprocess import run
from typing import Tuple

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.semparse.friction_q_util import get_explanation
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('friction-q-parser')
class FrictionQParserPredictor(Predictor):
    """
    Wrapper for the friction_q_semantic_parser model.
    """

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        Expects JSON that looks like ``{"question": "...", "table": "..."}``.
        """
        question_text = json_dict["question"]

        reader = self._dataset_reader

        # pylint: disable=protected-access
        world_extractions = None
        if reader._extract_world_entities and reader._replace_world_entities:
            world_extractions, _ = reader.get_world_extractions(json_dict)
            json_dict['world_extractions'] = world_extractions
            if world_extractions is not None:
                json_dict = reader._replace_stemmed_entities(json_dict, reader._stemmer)
                question_text = json_dict['question']
        fixed_question_text = reader._fix_question(question_text)
        tokenized_question = reader._tokenizer.tokenize(fixed_question_text.lower())  # type: ignore

        instance = self._dataset_reader.text_to_instance(question_text,  # type: ignore
                                                         tokenized_question=tokenized_question,
                                                         world_extractions=world_extractions)

        world_extractions_out = {"world1": "N/A", "world2": "N/A"}
        if world_extractions is not None:
            world_extractions_out.update(world_extractions)

        extra_info = {'question': json_dict['question'],
                      'question_tokens': tokenized_question,
                      "world_extractions": world_extractions_out}
        return instance, extra_info

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance, return_dict = self._json_to_instance(inputs)
        outputs = self._model.forward_on_instance(instance)
        answer_index = outputs['answer_index']
        if answer_index == 0:
            answer = "A"
        elif answer_index == 1:
            answer = "B"
        else:
            answer = "None"
        outputs['answer'] = answer

        return_dict.update(outputs)

        explanation = None
        if answer != "None":
            explanation = get_explanation(return_dict['logical_form'],
                                  return_dict['world_extractions'],
                                  answer_index)

        return_dict['explanation'] = explanation
        return sanitize(return_dict)
