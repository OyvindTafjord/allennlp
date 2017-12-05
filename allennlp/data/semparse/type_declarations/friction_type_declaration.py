"""
Defines all the types in the FrictionQuestions domain.
"""
from typing import Optional
from overrides import overrides

from nltk.sem.logic import Type, ComplexType, EntityType, ANY_TYPE

from allennlp.data.semparse.type_declarations.type_declaration import PlaceholderType, NamedBasicType, IdentityType

COMMON_NAME_MAPPING = {}

COMMON_TYPE_SIGNATURE = {}

def add_common_name_with_type(name, mapping, type_signature):
    COMMON_NAME_MAPPING[name] = mapping
    COMMON_TYPE_SIGNATURE[mapping] = type_signature

NUM_TYPE = NamedBasicType("NUM")
ATTR_TYPE = NamedBasicType("ATTR")
RDIR_TYPE = NamedBasicType("RDIR")
WORLD_TYPE = NamedBasicType("WORLD")

# Hardcoded flag to swap between two possible LF conventions
# (infer (qstate <attr> <rdir> <world>) (qstate <attr> <rdir> <world>) (qstate <attr> <rdir> <world>)) or
# (infer (<attr> <rdir> <world>) (<attr> <rdir> <world>) (<attr> <rdir> <world>))
USE_QSTATE_PREDICATE = False
if USE_QSTATE_PREDICATE:
    QSTATE_TYPE = NamedBasicType("QSTATE")
    #qstate: <ATTR, <QDIR, <WORLD, QSTATE>>>
    QSTATE_FUNCTION_TYPE = ComplexType(ATTR_TYPE,
                                       ComplexType(RDIR_TYPE,
                                                   ComplexType(WORLD_TYPE, QSTATE_TYPE)))
    #infer: <QSTATE, <QSTATE, <QSTATE, NUM>>>
    INFER_FUNCTION_TYPE = ComplexType(QSTATE_TYPE,
                                      ComplexType(QSTATE_TYPE,
                                                  ComplexType(QSTATE_TYPE, NUM_TYPE)))
    add_common_name_with_type("infer", "I10", INFER_FUNCTION_TYPE)
    add_common_name_with_type("qstate", "Q10", QSTATE_FUNCTION_TYPE)
    add_common_name_with_type("friction", "A10", ATTR_TYPE)
    add_common_name_with_type("smoothness", "A11", ATTR_TYPE)
    add_common_name_with_type("speed", "A12", ATTR_TYPE)
    add_common_name_with_type("heat", "A13", ATTR_TYPE)
    add_common_name_with_type("distance", "A14", ATTR_TYPE)
else:
    #attributes: <<QDIR, <WORLD, ATTR>>
    ATTR_FUNCTION_TYPE = ComplexType(RDIR_TYPE,
                                     ComplexType(WORLD_TYPE, ATTR_TYPE))

    #infer: <ATTR, <ATTR, <ATTR, NUM>>>
    INFER_FUNCTION_TYPE = ComplexType(ATTR_TYPE,
                                      ComplexType(ATTR_TYPE,
                                                  ComplexType(ATTR_TYPE, NUM_TYPE)))
    add_common_name_with_type("infer", "I10", INFER_FUNCTION_TYPE)
    add_common_name_with_type("friction", "A10", ATTR_FUNCTION_TYPE)
    add_common_name_with_type("smoothness", "A11", ATTR_FUNCTION_TYPE)
    add_common_name_with_type("speed", "A12", ATTR_FUNCTION_TYPE)
    add_common_name_with_type("heat", "A13", ATTR_FUNCTION_TYPE)
    add_common_name_with_type("distance", "A14", ATTR_FUNCTION_TYPE)

add_common_name_with_type("higher", "R10", RDIR_TYPE)
add_common_name_with_type("lower", "R11", RDIR_TYPE)
add_common_name_with_type("world1", "W11", WORLD_TYPE)
add_common_name_with_type("world2", "W12", WORLD_TYPE)
