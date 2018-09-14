import os
from subprocess import run
from typing import Tuple

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.semparse.friction_q_util import get_explanation, from_qr_spec_string, \
    words_from_entity_string, from_entity_cues_string
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
        tokenized_question = reader._tokenizer.tokenize(fixed_question_text.lower())

        qr_spec_override = None
        dynamic_entities = None
        if 'entitycues' in json_dict:
            entity_cues = from_entity_cues_string(json_dict['entitycues'])
            dynamic_entities = reader._dynamic_entities.copy()
            for entity, cues in entity_cues.items():
                key = "a:" + entity
                entity_strings = [words_from_entity_string(entity).lower()]
                entity_strings += cues
                dynamic_entities[key] = " ".join(entity_strings)

        if 'qrspec' in json_dict:
            qr_spec_override = from_qr_spec_string(json_dict['qrspec'])
            old_entities = dynamic_entities
            if old_entities is None:
                old_entities = reader._dynamic_entities.copy()
            dynamic_entities = {}
            for s in qr_spec_override:
                for k in s.keys():
                    key = "a:"+k
                    value = old_entities.get(key, words_from_entity_string(k).lower())
                    dynamic_entities[key] = value

        instance = self._dataset_reader.text_to_instance(question_text,
                                                         tokenized_question=tokenized_question,
                                                         world_extractions=world_extractions,
                                                         qr_spec_override=qr_spec_override,
                                                         dynamic_entities_override=dynamic_entities)

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
        world = instance.fields['world'].metadata
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
                                  answer_index,
                                  world)
        else:
            explanation =[{"header": "No consistent interpretation found!", "content": []}]

        return_dict['explanation'] = explanation
        return sanitize(return_dict)
