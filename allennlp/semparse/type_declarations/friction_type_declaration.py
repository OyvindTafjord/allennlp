"""
Defines all the types in the FrictionQuestions domain.
"""
from typing import Optional
from overrides import overrides

from nltk.sem.logic import Type, BasicType, ANY_TYPE, ComplexType as NltkComplexType

from allennlp.semparse.type_declarations.type_declaration import ComplexType, NamedBasicType

class FrictionTypeDeclaration:
    def __init__(self, syntax: str):

        self.COMMON_NAME_MAPPING = {}

        self.COMMON_TYPE_SIGNATURE = {}

        NUM_TYPE = NamedBasicType("NUM")
        ATTR_TYPE = NamedBasicType("ATTR")
        RDIR_TYPE = NamedBasicType("RDIR")
        WORLD_TYPE = NamedBasicType("WORLD")
        VAR_TYPE = NamedBasicType("VAR")

        self.BASIC_TYPES = {NUM_TYPE, ATTR_TYPE, RDIR_TYPE, WORLD_TYPE, VAR_TYPE}

        # Hack to expose it
        self.WORLD_TYPE = WORLD_TYPE

        # Flag to swap between three possible LF conventions
        # with_q_state: (infer (qstate <attr> <rdir> <world>) (qstate <attr> <rdir> <world>) (qstate <attr> <rdir> <world>))
        # with_variable: WIP
        # with_type_2: Default + (infer (and (<attr> <rdir> <world>)  (<attr> <rdir> <world>)) (<attr> <rdir> <world>) (<attr> <rdir> <world>))
        # with_type_2_split_q: infer takes only two arguments
        # Default: (infer (<attr> <rdir> <world>) (<attr> <rdir> <world>) (<attr> <rdir> <world>))

        if syntax == "with_q_state":
            QSTATE_TYPE = NamedBasicType("QSTATE")
            # qstate: <ATTR, <QDIR, <WORLD, QSTATE>>>
            QSTATE_FUNCTION_TYPE = ComplexType(ATTR_TYPE,
                                               ComplexType(RDIR_TYPE,
                                                           ComplexType(WORLD_TYPE, QSTATE_TYPE)))
            # infer: <QSTATE, <QSTATE, <QSTATE, NUM>>>
            INFER_FUNCTION_TYPE = ComplexType(QSTATE_TYPE,
                                              ComplexType(QSTATE_TYPE,
                                                          ComplexType(QSTATE_TYPE, NUM_TYPE)))
            self.add_common_name_with_type("infer", "I10", INFER_FUNCTION_TYPE)
            self.add_common_name_with_type("qstate", "Q10", QSTATE_FUNCTION_TYPE)
            self.add_common_name_with_type("friction", "A10", ATTR_TYPE)
            self.add_common_name_with_type("smoothness", "A11", ATTR_TYPE)
            self.add_common_name_with_type("speed", "A12", ATTR_TYPE)
            self.add_common_name_with_type("heat", "A13", ATTR_TYPE)
            self.add_common_name_with_type("distance", "A14", ATTR_TYPE)

            self.CURRIED_FUNCTIONS = {
                QSTATE_FUNCTION_TYPE: 3,
                INFER_FUNCTION_TYPE: 3
            }
        elif syntax == "with_variable":

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
            self.add_common_name_with_type("infer", "I10", INFER_FUNCTION_TYPE)
            self.add_common_name_with_type("friction", "A10", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("smoothness", "A11", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("speed", "A12", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("heat", "A13", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("distance", "A14", ATTR_FUNCTION_TYPE)

            self.add_common_name_with_type("worldvar", "V10", WORLDVAR_FUNCTION_TYPE)

            self.CURRIED_FUNCTIONS = {
                ATTR_FUNCTION_TYPE: 2,
                INFER_FUNCTION_TYPE: 5
            }
        elif syntax == "with_type_2" or syntax == "with_type_2_world_entities":
            # attributes: <<QDIR, <WORLD, ATTR>>
            ATTR_FUNCTION_TYPE = ComplexType(RDIR_TYPE,
                                             ComplexType(WORLD_TYPE, ATTR_TYPE))

            AND_FUNCTION_TYPE = ComplexType(ATTR_TYPE, ComplexType(ATTR_TYPE, ATTR_TYPE))

            # infer: <ATTR, <ATTR, <ATTR, NUM>>>
            INFER_FUNCTION_TYPE = ComplexType(ATTR_TYPE,
                                              ComplexType(ATTR_TYPE,
                                                          ComplexType(ATTR_TYPE, NUM_TYPE)))
            self.add_common_name_with_type("infer", "I10", INFER_FUNCTION_TYPE)
            self.add_common_name_with_type("friction", "A10", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("smoothness", "A11", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("speed", "A12", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("heat", "A13", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("distance", "A14", ATTR_FUNCTION_TYPE)

            # For simplicity we treat "high" and "low" as directions as well
            self.add_common_name_with_type("high", "R12", RDIR_TYPE)
            self.add_common_name_with_type("low", "R13", RDIR_TYPE)
            self.add_common_name_with_type("and", "C10", AND_FUNCTION_TYPE)

            self.CURRIED_FUNCTIONS = {
                ATTR_FUNCTION_TYPE: 2,
                INFER_FUNCTION_TYPE: 3,
                AND_FUNCTION_TYPE: 2
            }

        elif syntax == "with_type_2_general":
            # attributes: <<QDIR, <WORLD, ATTR>>
            ATTR_FUNCTION_TYPE = ComplexType(RDIR_TYPE,
                                             ComplexType(WORLD_TYPE, ATTR_TYPE))

            AND_FUNCTION_TYPE = ComplexType(ATTR_TYPE, ComplexType(ATTR_TYPE, ATTR_TYPE))

            # infer: <ATTR, <ATTR, <ATTR, NUM>>>
            INFER_FUNCTION_TYPE = ComplexType(ATTR_TYPE,
                                              ComplexType(ATTR_TYPE,
                                                          ComplexType(ATTR_TYPE, NUM_TYPE)))
            self.add_common_name_with_type("infer", "I10", INFER_FUNCTION_TYPE)
            self.add_common_name_with_type("friction", "A10", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("smoothness", "A11", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("speed", "A12", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("heat", "A13", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("distance", "A14", ATTR_FUNCTION_TYPE)
            # New attributes:
            self.add_common_name_with_type("acceleration", "A15", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("amountSweat", "A16", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("apparentSize", "A17", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("breakability", "A18", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("brightness", "A19", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("exerciseIntensity", "A20", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("flexibility", "A21", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("gravity", "A22", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("loudness", "A23", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("mass", "A24", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("strength", "A25", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("thickness", "A26", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("time", "A27", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("weight", "A28", ATTR_FUNCTION_TYPE)

            # For simplicity we treat "high" and "low" as directions as well
            self.add_common_name_with_type("high", "R12", RDIR_TYPE)
            self.add_common_name_with_type("low", "R13", RDIR_TYPE)
            self.add_common_name_with_type("and", "C10", AND_FUNCTION_TYPE)

            self.CURRIED_FUNCTIONS = {
                ATTR_FUNCTION_TYPE: 2,
                INFER_FUNCTION_TYPE: 3,
                AND_FUNCTION_TYPE: 2
            }

        elif syntax == "with_type_2_split_q":
            # attributes: <<QDIR, <WORLD, ATTR>>
            ATTR_FUNCTION_TYPE = ComplexType(RDIR_TYPE,
                                             ComplexType(WORLD_TYPE, ATTR_TYPE))

            AND_FUNCTION_TYPE = ComplexType(ATTR_TYPE, ComplexType(ATTR_TYPE, ATTR_TYPE))

            # infer: <ATTR, <ATTR, NUM>>
            INFER_FUNCTION_TYPE = ComplexType(ATTR_TYPE,
                                               ComplexType(ATTR_TYPE, NUM_TYPE))
            self.add_common_name_with_type("infer", "I10", INFER_FUNCTION_TYPE)
            self.add_common_name_with_type("friction", "A10", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("smoothness", "A11", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("speed", "A12", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("heat", "A13", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("distance", "A14", ATTR_FUNCTION_TYPE)

            self.add_common_name_with_type("high", "R12", RDIR_TYPE)
            self.add_common_name_with_type("low", "R13", RDIR_TYPE)
            self.add_common_name_with_type("and", "C10", AND_FUNCTION_TYPE)

            self.CURRIED_FUNCTIONS = {
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
            self.add_common_name_with_type("infer", "I10", INFER_FUNCTION_TYPE)
            self.add_common_name_with_type("friction", "A10", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("smoothness", "A11", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("speed", "A12", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("heat", "A13", ATTR_FUNCTION_TYPE)
            self.add_common_name_with_type("distance", "A14", ATTR_FUNCTION_TYPE)

            self.CURRIED_FUNCTIONS = {
                ATTR_FUNCTION_TYPE: 2,
                INFER_FUNCTION_TYPE: 3
            }

        self.add_common_name_with_type("higher", "R10", RDIR_TYPE)
        self.add_common_name_with_type("lower", "R11", RDIR_TYPE)

        if syntax != "with_type_2_world_entities":
            self.add_common_name_with_type("world1", "W11", WORLD_TYPE)
            self.add_common_name_with_type("world2", "W12", WORLD_TYPE)

        self.STARTING_TYPES = [NUM_TYPE]

    def add_common_name_with_type(self, name, mapping, type_signature):
        self.COMMON_NAME_MAPPING[name] = mapping
        self.COMMON_TYPE_SIGNATURE[mapping] = type_signature