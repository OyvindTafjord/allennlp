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
        elif syntax == "with_type_2_attr_entities" or syntax == "with_type_2_general_attr_entities":
            # attributes: <<QDIR, <WORLD, ATTR>>
            ATTR_FUNCTION_TYPE = ComplexType(RDIR_TYPE,
                                             ComplexType(WORLD_TYPE, ATTR_TYPE))

            AND_FUNCTION_TYPE = ComplexType(ATTR_TYPE, ComplexType(ATTR_TYPE, ATTR_TYPE))

            # infer: <ATTR, <ATTR, <ATTR, NUM>>>
            INFER_FUNCTION_TYPE = ComplexType(ATTR_TYPE,
                                              ComplexType(ATTR_TYPE,
                                                          ComplexType(ATTR_TYPE, NUM_TYPE)))
            self.add_common_name_with_type("infer", "I10", INFER_FUNCTION_TYPE)
            self.add_common_name_with_type("fakeattr", "A99", ATTR_FUNCTION_TYPE)

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

        # Hack to expose types
        self.WORLD_TYPE = WORLD_TYPE
        self.ATTR_FUNCTION_TYPE = ATTR_FUNCTION_TYPE

        self.STARTING_TYPES = [NUM_TYPE]

        if syntax == "life_cycle":
            ORGANISM_TYPE = NamedBasicType("ORGANISM")
            STAGE_TYPE = NamedBasicType("STAGE")
            STAGEIDX_TYPE = NamedBasicType("STAGEIDX")

            ORGANISM_FUNCTION_TYPE = ComplexType(ORGANISM_TYPE, NUM_TYPE)

            ORGANISM_STAGE_FUNCTION_TYPE = ComplexType(ORGANISM_TYPE,
                                                 ComplexType(STAGE_TYPE, NUM_TYPE))

            ORGANISM_STAGE2_FUNCTION_TYPE = ComplexType(ORGANISM_TYPE,
                                                ComplexType(STAGE_TYPE,
                                                    ComplexType(STAGE_TYPE, NUM_TYPE)))

            ORGANISM_STAGEIDX_FUNCTION_TYPE = ComplexType(ORGANISM_TYPE,
                                                       ComplexType(STAGEIDX_TYPE, NUM_TYPE))

            self.CURRIED_FUNCTIONS = {
                ORGANISM_FUNCTION_TYPE: 1,
                ORGANISM_STAGE_FUNCTION_TYPE: 2,
                ORGANISM_STAGEIDX_FUNCTION_TYPE: 2,
                ORGANISM_STAGE2_FUNCTION_TYPE: 3
            }

            self.add_common_name_with_type("qCorrectlyOrdered", "F10", ORGANISM_FUNCTION_TYPE)
            self.add_common_name_with_type("qCountStages", "F11", ORGANISM_FUNCTION_TYPE)
            self.add_common_name_with_type("qIsNotAStageOf", "F12", ORGANISM_FUNCTION_TYPE)
            self.add_common_name_with_type("qIsAStageOf", "F13", ORGANISM_FUNCTION_TYPE)
            self.add_common_name_with_type("qStageBefore", "G10", ORGANISM_STAGE_FUNCTION_TYPE)
            self.add_common_name_with_type("qNextStage", "G11", ORGANISM_STAGE_FUNCTION_TYPE)
            self.add_common_name_with_type("qStageBetween", "H10", ORGANISM_STAGE2_FUNCTION_TYPE)
            self.add_common_name_with_type("qStageAt", "J10", ORGANISM_STAGEIDX_FUNCTION_TYPE)

            self.add_common_name_with_type("african_goliath_beetle", "O10", ORGANISM_TYPE)
            self.add_common_name_with_type("bear", "O11", ORGANISM_TYPE)
            self.add_common_name_with_type("bird", "O12", ORGANISM_TYPE)
            self.add_common_name_with_type("black_widow", "O13", ORGANISM_TYPE)
            self.add_common_name_with_type("butterfly", "O14", ORGANISM_TYPE)
            self.add_common_name_with_type("chicken", "O15", ORGANISM_TYPE)
            self.add_common_name_with_type("cockroach", "O16", ORGANISM_TYPE)
            self.add_common_name_with_type("crickets", "O17", ORGANISM_TYPE)
            self.add_common_name_with_type("crocodile", "O18", ORGANISM_TYPE)
            self.add_common_name_with_type("darkling_beetle", "O19", ORGANISM_TYPE)
            self.add_common_name_with_type("dragonflies", "O20", ORGANISM_TYPE)
            self.add_common_name_with_type("fish", "O21", ORGANISM_TYPE)
            self.add_common_name_with_type("fleas", "O22", ORGANISM_TYPE)
            self.add_common_name_with_type("frog", "O23", ORGANISM_TYPE)
            self.add_common_name_with_type("grasshopper", "O24", ORGANISM_TYPE)
            self.add_common_name_with_type("hammerhead_shark", "O25", ORGANISM_TYPE)
            self.add_common_name_with_type("hen", "O26", ORGANISM_TYPE)
            self.add_common_name_with_type("human", "O27", ORGANISM_TYPE)
            self.add_common_name_with_type("lice", "O28", ORGANISM_TYPE)
            self.add_common_name_with_type("lizard", "O29", ORGANISM_TYPE)
            self.add_common_name_with_type("longleaf_pine", "O30", ORGANISM_TYPE)
            self.add_common_name_with_type("mosquito", "O31", ORGANISM_TYPE)
            self.add_common_name_with_type("moth", "O32", ORGANISM_TYPE)
            self.add_common_name_with_type("newt", "O33", ORGANISM_TYPE)
            self.add_common_name_with_type("oviaporous_snake", "O34", ORGANISM_TYPE)
            self.add_common_name_with_type("penguin", "O35", ORGANISM_TYPE)
            self.add_common_name_with_type("plant", "O36", ORGANISM_TYPE)
            self.add_common_name_with_type("praying_mantises", "O37", ORGANISM_TYPE)
            self.add_common_name_with_type("salmon", "O38", ORGANISM_TYPE)
            self.add_common_name_with_type("scorpion", "O39", ORGANISM_TYPE)
            self.add_common_name_with_type("spider", "O40", ORGANISM_TYPE)
            self.add_common_name_with_type("toad", "O41", ORGANISM_TYPE)
            self.add_common_name_with_type("tree", "O42", ORGANISM_TYPE)
            self.add_common_name_with_type("whale", "O43", ORGANISM_TYPE)
            self.add_common_name_with_type("wolf", "O44", ORGANISM_TYPE)

            self.add_common_name_with_type("adolescent", "S10", STAGE_TYPE)
            self.add_common_name_with_type("adult", "S11", STAGE_TYPE)
            self.add_common_name_with_type("adulthood", "S12", STAGE_TYPE)
            self.add_common_name_with_type("alevin", "S13", STAGE_TYPE)
            self.add_common_name_with_type("baby", "S14", STAGE_TYPE)
            self.add_common_name_with_type("big_chick", "S15", STAGE_TYPE)
            self.add_common_name_with_type("bottle_brush", "S16", STAGE_TYPE)
            self.add_common_name_with_type("caterpillar", "S17", STAGE_TYPE)
            self.add_common_name_with_type("chick", "S18", STAGE_TYPE)
            self.add_common_name_with_type("childhood", "S19", STAGE_TYPE)
            self.add_common_name_with_type("cocoon", "S20", STAGE_TYPE)
            self.add_common_name_with_type("decline", "S21", STAGE_TYPE)
            self.add_common_name_with_type("eft", "S22", STAGE_TYPE)
            self.add_common_name_with_type("egg", "S23", STAGE_TYPE)
            self.add_common_name_with_type("embryo", "S24", STAGE_TYPE)
            self.add_common_name_with_type("fledgling", "S25", STAGE_TYPE)
            self.add_common_name_with_type("foetus", "S26", STAGE_TYPE)
            self.add_common_name_with_type("froglet", "S27", STAGE_TYPE)
            self.add_common_name_with_type("fry", "S28", STAGE_TYPE)
            self.add_common_name_with_type("full_grown_tree", "S29", STAGE_TYPE)
            self.add_common_name_with_type("gestation", "S30", STAGE_TYPE)
            self.add_common_name_with_type("grass", "S31", STAGE_TYPE)
            self.add_common_name_with_type("hatchling", "S32", STAGE_TYPE)
            self.add_common_name_with_type("juvenile", "S33", STAGE_TYPE)
            self.add_common_name_with_type("larva", "S34", STAGE_TYPE)
            self.add_common_name_with_type("mature", "S35", STAGE_TYPE)
            self.add_common_name_with_type("matured_tree", "S36", STAGE_TYPE)
            self.add_common_name_with_type("nestling", "S37", STAGE_TYPE)
            self.add_common_name_with_type("nymph", "S38", STAGE_TYPE)
            self.add_common_name_with_type("parr", "S39", STAGE_TYPE)
            self.add_common_name_with_type("prenatal", "S40", STAGE_TYPE)
            self.add_common_name_with_type("pup", "S41", STAGE_TYPE)
            self.add_common_name_with_type("pupa", "S42", STAGE_TYPE)
            self.add_common_name_with_type("sapling", "S43", STAGE_TYPE)
            self.add_common_name_with_type("seed", "S44", STAGE_TYPE)
            self.add_common_name_with_type("seedling", "S45", STAGE_TYPE)
            self.add_common_name_with_type("smolt", "S46", STAGE_TYPE)
            self.add_common_name_with_type("spawner", "S47", STAGE_TYPE)
            self.add_common_name_with_type("spiderling", "S48", STAGE_TYPE)
            self.add_common_name_with_type("sprout", "S49", STAGE_TYPE)
            self.add_common_name_with_type("subadult", "S50", STAGE_TYPE)
            self.add_common_name_with_type("tadpole", "S51", STAGE_TYPE)
            self.add_common_name_with_type("tadpole_with_legs", "S52", STAGE_TYPE)
            self.add_common_name_with_type("teenage", "S53", STAGE_TYPE)
            self.add_common_name_with_type("toadlet", "S54", STAGE_TYPE)
            self.add_common_name_with_type("young", "S55", STAGE_TYPE)
            self.add_common_name_with_type("young_adult", "S56", STAGE_TYPE)

            self.add_common_name_with_type("1", "X10", STAGEIDX_TYPE)
            self.add_common_name_with_type("2", "X11", STAGEIDX_TYPE)
            self.add_common_name_with_type("3", "X12", STAGEIDX_TYPE)
            self.add_common_name_with_type("4", "X13", STAGEIDX_TYPE)
            self.add_common_name_with_type("5", "X14", STAGEIDX_TYPE)
            self.add_common_name_with_type("6", "X15", STAGEIDX_TYPE)
            self.add_common_name_with_type("7", "X16", STAGEIDX_TYPE)
            self.add_common_name_with_type("last", "X17", STAGEIDX_TYPE)
            self.add_common_name_with_type("middle", "X18", STAGEIDX_TYPE)

            self.BASIC_TYPES = {NUM_TYPE, ORGANISM_TYPE, STAGE_TYPE, STAGEIDX_TYPE}

    def add_common_name_with_type(self, name, mapping, type_signature):
        self.COMMON_NAME_MAPPING[name] = mapping
        self.COMMON_TYPE_SIGNATURE[mapping] = type_signature