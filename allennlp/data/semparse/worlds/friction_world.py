"""
This module defines FrictionWorld, with a simple domain theory for qualitative reasoning about
friction.
"""

from collections import defaultdict
import operator
from typing import Any, List, Dict, Set, Callable, TypeVar, Union
from overrides import overrides
import pyparsing

from allennlp.common import util
from allennlp.common.util import JsonDict
from allennlp.data.semparse.type_declarations.friction_type_declaration import (COMMON_NAME_MAPPING,
                                                                            COMMON_TYPE_SIGNATURE)
from allennlp.data.semparse.worlds.world import World

AttributeType = TypeVar('AttributeType', str, int)  # pylint: disable=invalid-name


class FrictionWorld(World):
    """
    Class defining the friction domain theory world.

    Parameters
    ----------
    """
    # pylint: disable=too-many-public-methods
    def __init__(self) -> None:
        super(FrictionWorld, self).__init__(global_type_signatures=COMMON_TYPE_SIGNATURE,
                                            global_name_mapping=COMMON_NAME_MAPPING)

    @overrides
    def _map_name(self, name: str) -> str:
        return COMMON_NAME_MAPPING[name] if name in COMMON_NAME_MAPPING else name
