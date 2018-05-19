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
from allennlp.semparse.type_declarations.friction_type_declaration import FrictionTypeDeclaration
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

    def __init__(self, table_graph: TableQuestionKnowledgeGraph, syntax: str = "WithType2") -> None:
        self.types = FrictionTypeDeclaration(syntax)
        super(FrictionWorld, self).__init__(
                                            global_type_signatures=self.types.COMMON_TYPE_SIGNATURE,
                                            global_name_mapping=self.types.COMMON_NAME_MAPPING)
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
        translated_name = name
        if name in self.types.COMMON_NAME_MAPPING:
            translated_name = self.types.COMMON_NAME_MAPPING[name]
        elif name in self.local_name_mapping:
            translated_name = self.local_name_mapping[name]
        elif name.startswith("world"):
            translated_name = "W1"+name[-1]
            self._add_name_mapping(name, translated_name, self.types.WORLD_TYPE)
        return translated_name

    def _get_curried_functions(self) -> Dict[Type, int]:
        return self.types.CURRIED_FUNCTIONS

    @overrides
    def get_basic_types(self) -> Set[Type]:
        return self.types.BASIC_TYPES

    @overrides
    def get_valid_starting_types(self) -> Set[Type]:
        return self.types.STARTING_TYPES

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

    # Size translation for absolute and relative values
    qr_size = {
        'higher': 1,
        'high': 1,
        'low': -1,
        'lower': -1
    }

    @staticmethod
    def check_compatible(setup: List, answer: List) -> bool:
        attribute_dir = FrictionWorld.qr_coeff[setup[0]] * FrictionWorld.qr_coeff[answer[0]]
        change_same = 1 if FrictionWorld.qr_size[setup[1]] == FrictionWorld.qr_size[answer[1]] else -1
        world_same = 1 if setup[2] == answer[2] else -1
        return attribute_dir * change_same * world_same == 1

    @staticmethod
    def exec_infer(setup, *answers):
        answer_index = -1
        if len(answers) == 1:
            if FrictionWorld.check_compatible(setup, answers[0]):
                return 1
            else:
                return 0
        for index, answer in enumerate(answers):
            if FrictionWorld.check_compatible(setup, answer):
                if answer_index > -1:
                    # Found two valid answers
                    answer_index = -2
                else:
                    answer_index = index
        return answer_index

    @staticmethod
    def exec_and(expr):
        if len(expr) == 0 or expr[0] != 'and':
            return expr
        args = expr[1:]
        if len(args) == 1:
            return args[0]
        if len(args) > 2:
            # More than 2 arguments not allowed by current grammar
            return None
        if FrictionWorld.check_compatible(args[0], args[1]):
            # Check that arguments are compatible, then fine to keep just one
            return args[0]
        return None

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
            args = [FrictionWorld.exec_and(arg) for arg in parse[0][1:]]
            if None in args:
                return -1
            return FrictionWorld.exec_infer(*args)
        return -1
