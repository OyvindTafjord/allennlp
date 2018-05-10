"""
This module defines FrictionWorld, with a simple domain theory for qualitative reasoning about
friction.
"""

from collections import defaultdict
import operator
from typing import Any, List, Dict, Set, Callable, TypeVar, Union
from nltk.sem.logic import Type
from overrides import overrides
import pyparsing

from allennlp.semparse import util as semparse_util
from allennlp.semparse.type_declarations import friction_type_declaration as types
from allennlp.semparse.worlds.world import World
from allennlp.semparse.contexts import TableQuestionKnowledgeGraph

AttributeType = TypeVar('AttributeType', str, int)  # pylint: disable=invalid-name


class FrictionWorld(World):
    """
    Class defining the friction domain theory world.

    Parameters
    ----------
    """
    # pylint: disable=too-many-public-methods

    def __init__(self, table_graph: TableQuestionKnowledgeGraph) -> None:
        super(FrictionWorld, self).__init__(
                                              global_type_signatures=types.COMMON_TYPE_SIGNATURE,
                                              global_name_mapping=types.COMMON_NAME_MAPPING)
        self.table_graph = table_graph

        # For every new Sempre column name seen, we update this counter to map it to a new NLTK name.
        self._column_counter = 0

        # This adds all of the cell and column names to our local name mapping, including null
        # cells and columns and a few simple numbers, so we can get them as valid actions in the
        # parser.  The null cell and column are used to check against empty sets, e.g., for
        # questions like "Is there a team that won three times in a row?".
        for entity in table_graph.entities: #+ ['fb:cell.null', 'fb:row.row.null', '-1', '0', '1']:
            self._map_name(entity, keep_mapping=True)

        self._entity_set = set(table_graph.entities)

    def is_table_entity(self, entity_name: str) -> bool:
        """
        Returns ``True`` if the given entity is one of the entities in the table.
        """
        return entity_name in self._entity_set

    @overrides
    def _map_name(self, name: str, keep_mapping: bool = False) -> str:
        return types.COMMON_NAME_MAPPING[name] if name in types.COMMON_NAME_MAPPING else name

    def _get_curried_functions(self) -> Dict[Type, int]:
        return types.CURRIED_FUNCTIONS

    @overrides
    def get_basic_types(self) -> Set[Type]:
        return types.BASIC_TYPES

    @overrides
    def get_valid_starting_types(self) -> Set[Type]:
        return types.STARTING_TYPES

    def is_table_entity(self, entity_name: str) -> bool:
        """
        Returns ``True`` if the given entity is one of the entities in the table.
        """
        return False

    # Simple table for how attributes relates qualitatively to friction
    qr_coeff = {
        "friction": 1,
        "smoothness": -1,
        "speed": -1,
        "distance": -1,
        "heat": 1
    }

    @staticmethod
    def _exec_infer(setup, *answers):
        answer_index = -1
        for index, answer in enumerate(answers):
            attribute_dir = FrictionWorld.qr_coeff[setup[0]] * FrictionWorld.qr_coeff[answer[0]]
            change_same = 1 if setup[1] == answer[1] else -1
            world_same = 1 if setup[2] == answer[2] else -1
            if attribute_dir * change_same * world_same == 1:
                if answer_index > -1:
                    # Found two valid answers
                    answer_index = -2
                else:
                    answer_index = index
        return answer_index

    @staticmethod
    def execute(lf: str) -> int:
        """
        Very basic model for executing friction logical forms. For now returns answer index (or
        -1 if no answer can be concluded)
        """
        parse = semparse_util.lisp_to_nested_expression(lf)
        if len(parse) < 1 and len(parse[0]) < 2:
            return -1
        if parse[0][0] == 'infer':
            return FrictionWorld._exec_infer(*parse[0][1:])
        return -1
