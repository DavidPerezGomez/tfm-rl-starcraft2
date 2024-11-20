from enum import IntEnum


class AllActions(IntEnum):
    NO_OP = 0

    # Resource manager actions
    HARVEST_MINERALS = 11
    # COLLECT_GAS      = 12
    # BUILD_REFINERY   = 13

    # Base Manager actions
    RECRUIT_SCV_0        = 21
    RECRUIT_SCV_1        = 22
    RECRUIT_SCV_2        = 23
    BUILD_SUPPLY_DEPOT   = 24
    BUILD_COMMAND_CENTER = 25

    # ArmyRecruit Manager actions
    BUILD_BARRACKS = 31
    RECRUIT_MARINE = 32
    # BUILD_TECH_LAB = 33
    # RECRUIT_DOCTOR = 34

    # ArmyAttack Manager actions
    ATTACK_BUILDING_WITH_SINGLE_UNIT    = 41
    ATTACK_WORKER_WITH_SINGLE_UNIT      = 42
    ATTACK_ARMY_WITH_SINGLE_UNIT        = 43
    ATTACK_BUILDING_WITH_ENTIRE_ARMY    = 44
    ATTACK_WORKER_WITH_ENTIRE_ARMY      = 45
    ATTACK_ARMY_WITH_ENTIRE_ARMY        = 46
    # ATTACK_WITH_SQUAD_5       = 42
    # ATTACK_WITH_SQUAD_10      = 43
    # ATTACK_WITH_SQUAD_15      = 44
    # ATTACK_WITH_FULL_ARMY     = 45

class GameManagerActions(IntEnum):
    # GATHER_RESOURCES = 10
    EXPAND_BASE = 20
    EXPAND_ARMY = 30
    ATTACK = 40

# class ResourceManagerActions(IntEnum):
#     NO_OP            = AllActions.NO_OP
#     HARVEST_MINERALS = AllActions.HARVEST_MINERALS
#     # COLLECT_GAS      = AllActions.COLLECT_GAS
#     # BUILD_REFINERY   = AllActions.BUILD_REFINERY

class BaseManagerActions(IntEnum):
    NO_OP                = AllActions.NO_OP
    HARVEST_MINERALS     = AllActions.HARVEST_MINERALS
    RECRUIT_SCV_0          = AllActions.RECRUIT_SCV_0
    RECRUIT_SCV_1          = AllActions.RECRUIT_SCV_1
    RECRUIT_SCV_2          = AllActions.RECRUIT_SCV_2
    BUILD_SUPPLY_DEPOT   = AllActions.BUILD_SUPPLY_DEPOT
    BUILD_COMMAND_CENTER = AllActions.BUILD_COMMAND_CENTER

class ArmyRecruitManagerActions(IntEnum):
    NO_OP          = AllActions.NO_OP
    HARVEST_MINERALS     = AllActions.HARVEST_MINERALS
    # RECRUIT_SCV_0        = AllActions.RECRUIT_SCV_0
    BUILD_SUPPLY_DEPOT   = AllActions.BUILD_SUPPLY_DEPOT
    BUILD_BARRACKS = AllActions.BUILD_BARRACKS
    RECRUIT_MARINE = AllActions.RECRUIT_MARINE
    # RECRUIT_DOCTOR = AllActions.RECRUIT_DOCTOR
    # BUILD_TECH_LAB = AllActions.BUILD_TECH_LAB

class ArmyAttackManagerActions(IntEnum):
    NO_OP                               = AllActions.NO_OP
    ATTACK_BUILDING_WITH_SINGLE_UNIT    = AllActions.ATTACK_BUILDING_WITH_SINGLE_UNIT
    ATTACK_WORKER_WITH_SINGLE_UNIT      = AllActions.ATTACK_WORKER_WITH_SINGLE_UNIT
    ATTACK_ARMY_WITH_SINGLE_UNIT        = AllActions.ATTACK_ARMY_WITH_SINGLE_UNIT
    ATTACK_BUILDING_WITH_ENTIRE_ARMY    = AllActions.ATTACK_BUILDING_WITH_ENTIRE_ARMY
    ATTACK_WORKER_WITH_ENTIRE_ARMY      = AllActions.ATTACK_WORKER_WITH_ENTIRE_ARMY
    ATTACK_ARMY_WITH_ENTIRE_ARMY        = AllActions.ATTACK_ARMY_WITH_ENTIRE_ARMY
    # ATTACK_WITH_SQUAD_5       = AllActions.ATTACK_WITH_SQUAD_5
    # ATTACK_WITH_SQUAD_10      = AllActions.ATTACK_WITH_SQUAD_10
    # ATTACK_WITH_SQUAD_15      = AllActions.ATTACK_WITH_SQUAD_15
    # ATTACK_WITH_FULL_ARMY     = AllActions.ATTACK_WITH_FULL_ARMY
