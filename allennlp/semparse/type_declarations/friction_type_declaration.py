"""
Defines all the types in the FrictionQuestions domain.
"""
from typing import Optional
from overrides import overrides

from nltk.sem.logic import Type, BasicType, ANY_TYPE, ComplexType as NltkComplexType

from allennlp.semparse.type_declarations.type_declaration import ComplexType, NamedBasicType

COMMON_NAME_MAPPING = {}

COMMON_TYPE_SIGNATURE = {}


def add_common_name_with_type(name, mapping, type_signature):
    COMMON_NAME_MAPPING[name] = mapping
    COMMON_TYPE_SIGNATURE[mapping] = type_signature

NUM_TYPE = NamedBasicType("NUM")
ATTR_TYPE = NamedBasicType("ATTR")
RDIR_TYPE = NamedBasicType("RDIR")
WORLD_TYPE = NamedBasicType("WORLD")
VAR_TYPE = NamedBasicType("VAR")

BASIC_TYPES = {NUM_TYPE, ATTR_TYPE, RDIR_TYPE, WORLD_TYPE, VAR_TYPE}

# Hardcoded flag to swap between three possible LF conventions
# WithQState: (infer (qstate <attr> <rdir> <world>) (qstate <attr> <rdir> <world>) (qstate <attr> <rdir> <world>))
# WithVar:
# Default: (infer (<attr> <rdir> <world>) (<attr> <rdir> <world>) (<attr> <rdir> <world>))
LOGICAL_FORM_SYNTAX = "WithType2"
LOGICAL_FORM_SYNTAX = "WithType2SplitQ"
# LOGICAL_FORM_SYNTAX = "Default"

if LOGICAL_FORM_SYNTAX == "WithQState":
    QSTATE_TYPE = NamedBasicType("QSTATE")
    # qstate: <ATTR, <QDIR, <WORLD, QSTATE>>>
    QSTATE_FUNCTION_TYPE = ComplexType(ATTR_TYPE,
                                       ComplexType(RDIR_TYPE,
                                                   ComplexType(WORLD_TYPE, QSTATE_TYPE)))
    # infer: <QSTATE, <QSTATE, <QSTATE, NUM>>>
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

    CURRIED_FUNCTIONS = {
        QSTATE_FUNCTION_TYPE: 3,
        INFER_FUNCTION_TYPE: 3
    }
elif LOGICAL_FORM_SYNTAX == "WithVariable":

    # worldvar: <WORLD, VAR>
    WORLDVAR_FUNCTION_TYPE = ComplexType(WORLD_TYPE, VAR_TYPE)
    # attributes: <<QDIR, <WORLD, ATTR>>
    ATTR_FUNCTION_TYPE = ComplexType(RDIR_TYPE,
                                     ComplexType(WORLD_TYPE, ATTR_TYPE))

    # infer: <ATTR, <ATTR, <ATTR, NUM>>>
    INFER_FUNCTION_TYPE = \
        ComplexType(VAR_TYPE,
                    ComplexType(VAR_TYPE,
                                ComplexType(ATTR_TYPE,
                                            ComplexType(ATTR_TYPE,
                                                        ComplexType(ATTR_TYPE, NUM_TYPE)))))
    add_common_name_with_type("infer", "I10", INFER_FUNCTION_TYPE)
    add_common_name_with_type("friction", "A10", ATTR_FUNCTION_TYPE)
    add_common_name_with_type("smoothness", "A11", ATTR_FUNCTION_TYPE)
    add_common_name_with_type("speed", "A12", ATTR_FUNCTION_TYPE)
    add_common_name_with_type("heat", "A13", ATTR_FUNCTION_TYPE)
    add_common_name_with_type("distance", "A14", ATTR_FUNCTION_TYPE)

    add_common_name_with_type("worldvar", "V10", WORLDVAR_FUNCTION_TYPE)

    CURRIED_FUNCTIONS = {
        ATTR_FUNCTION_TYPE: 2,
        INFER_FUNCTION_TYPE: 5
    }
elif LOGICAL_FORM_SYNTAX == "WithType2":
    # attributes: <<QDIR, <WORLD, ATTR>>
    ATTR_FUNCTION_TYPE = ComplexType(RDIR_TYPE,
                                     ComplexType(WORLD_TYPE, ATTR_TYPE))

    AND_FUNCTION_TYPE = ComplexType(ATTR_TYPE, ComplexType(ATTR_TYPE, ATTR_TYPE))

    # infer: <ATTR, <ATTR, <ATTR, NUM>>>
    INFER_FUNCTION_TYPE = ComplexType(ATTR_TYPE,
                                      ComplexType(ATTR_TYPE,
                                                  ComplexType(ATTR_TYPE, NUM_TYPE)))
    add_common_name_with_type("infer", "I10", INFER_FUNCTION_TYPE)
    add_common_name_with_type("friction", "A10", ATTR_FUNCTION_TYPE)
    add_common_name_with_type("smoothness", "A11", ATTR_FUNCTION_TYPE)
    add_common_name_with_type("speed", "A12", ATTR_FUNCTION_TYPE)
    add_common_name_with_type("heat", "A13", ATTR_FUNCTION_TYPE)
    add_common_name_with_type("distance", "A14", ATTR_FUNCTION_TYPE)

    add_common_name_with_type("high", "R12", RDIR_TYPE)
    add_common_name_with_type("low", "R13", RDIR_TYPE)
    add_common_name_with_type("and", "C10", AND_FUNCTION_TYPE)

    CURRIED_FUNCTIONS = {
        ATTR_FUNCTION_TYPE: 2,
        INFER_FUNCTION_TYPE: 3,
        AND_FUNCTION_TYPE: 2
    }

elif LOGICAL_FORM_SYNTAX == "WithType2SplitQ":
    # attributes: <<QDIR, <WORLD, ATTR>>
    ATTR_FUNCTION_TYPE = ComplexType(RDIR_TYPE,
                                     ComplexType(WORLD_TYPE, ATTR_TYPE))

    AND_FUNCTION_TYPE = ComplexType(ATTR_TYPE, ComplexType(ATTR_TYPE, ATTR_TYPE))

    # infer: <ATTR, <ATTR, NUM>>
    INFER_FUNCTION_TYPE = ComplexType(ATTR_TYPE,
                                       ComplexType(ATTR_TYPE, NUM_TYPE))
    add_common_name_with_type("infer", "I10", INFER_FUNCTION_TYPE)
    add_common_name_with_type("friction", "A10", ATTR_FUNCTION_TYPE)
    add_common_name_with_type("smoothness", "A11", ATTR_FUNCTION_TYPE)
    add_common_name_with_type("speed", "A12", ATTR_FUNCTION_TYPE)
    add_common_name_with_type("heat", "A13", ATTR_FUNCTION_TYPE)
    add_common_name_with_type("distance", "A14", ATTR_FUNCTION_TYPE)

    add_common_name_with_type("high", "R12", RDIR_TYPE)
    add_common_name_with_type("low", "R13", RDIR_TYPE)
    add_common_name_with_type("and", "C10", AND_FUNCTION_TYPE)

    CURRIED_FUNCTIONS = {
        ATTR_FUNCTION_TYPE: 2,
        INFER_FUNCTION_TYPE: 2,
        AND_FUNCTION_TYPE: 2
    }

else:
    # attributes: <<QDIR, <WORLD, ATTR>>
    ATTR_FUNCTION_TYPE = ComplexType(RDIR_TYPE,
                                     ComplexType(WORLD_TYPE, ATTR_TYPE))

    # infer: <ATTR, <ATTR, <ATTR, NUM>>>
    INFER_FUNCTION_TYPE = ComplexType(ATTR_TYPE,
                                      ComplexType(ATTR_TYPE,
                                                  ComplexType(ATTR_TYPE, NUM_TYPE)))
    add_common_name_with_type("infer", "I10", INFER_FUNCTION_TYPE)
    add_common_name_with_type("friction", "A10", ATTR_FUNCTION_TYPE)
    add_common_name_with_type("smoothness", "A11", ATTR_FUNCTION_TYPE)
    add_common_name_with_type("speed", "A12", ATTR_FUNCTION_TYPE)
    add_common_name_with_type("heat", "A13", ATTR_FUNCTION_TYPE)
    add_common_name_with_type("distance", "A14", ATTR_FUNCTION_TYPE)

    CURRIED_FUNCTIONS = {
        ATTR_FUNCTION_TYPE: 2,
        INFER_FUNCTION_TYPE: 3
    }

add_common_name_with_type("higher", "R10", RDIR_TYPE)
add_common_name_with_type("lower", "R11", RDIR_TYPE)
add_common_name_with_type("world1", "W11", WORLD_TYPE)
add_common_name_with_type("world2", "W12", WORLD_TYPE)


STARTING_TYPES = [NUM_TYPE]
