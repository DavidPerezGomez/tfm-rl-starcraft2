
from pysc2.env import sc2_env
from pysc2.lib import features, units
from tfm_sc2.actions import AllActions, ArmyAttackManagerActions, BaseManagerActions

SC2_CONFIG = dict(
    agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=256, minimap=64),
                    use_raw_units=True,
                    use_raw_actions=True),
    step_mul=32,#32,#48 (16 game steps per second)
    game_steps_per_episode=0,
    visualize=False,
    disable_fog=True
)
    # players=[sc2_env.Agent(sc2_env.Race.terran),
    #          sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy),],
    # agent_interface_format=features.AgentInterfaceFormat(
    #     action_space=actions.ActionSpace.RAW,
    #     use_raw_units=True,
    #     raw_resolution=256),
    # # This gives ~150 APM - a value of 8 would give ~300APM
    # step_mul=16,
    # # 0 - Let the game run for as long as necessary
    # game_steps_per_episode=0,
    # # Optional, but useful if we want to watch the game
    # visualize=True,
    # disable_fog=True)


MAP_CONFIGS = dict(
    Simple64=dict(
        map_name="Simple64",
        positions={
            "top_left": {
                units.Terran.CommandCenter: [(26, 35), (24, 72), (56, 31)],
                units.Terran.SupplyDepot:
                    [(21, 42), (21, 44), (22, 40), (22, 42), (22, 44), (22, 46), (24, 40), (24, 42), (24, 44), (24, 46),
                     (26, 40), (26, 42), (26, 44), (26, 46), (28, 40), (28, 42), (28, 44), (28, 46), (30, 40), (30, 42),
                     (30, 44), (32, 40), (32, 42), (32, 44)], # 24 supply depots
                units.Terran.Barracks:
                    [(32, 28), (35, 39), (36, 28), (38, 39)],
            },
            "bottom_right": {
                units.Terran.CommandCenter: [(54, 68), (24, 72), (56, 31)],
                units.Terran.SupplyDepot:
                    [(48, 60), (48, 62), (48, 64), (50, 60), (50, 62), (50, 64), (52, 58), (52, 60), (52, 62), (52, 64),
                     (54, 58), (54, 60), (54, 62), (54, 64), (55, 58), (55, 60), (55, 62), (55, 64), (57, 58), (57, 60),
                     (57, 62), (57, 64), (59, 60), (59, 62)], # 24 supply depots
                units.Terran.Barracks:
                    [(41, 64), (44, 75), (45, 64), (48, 75)],
            }
        },
        multiple_positions=True,
        players=[
            sc2_env.Agent(sc2_env.Race.terran),
            sc2_env.Agent(sc2_env.Race.terran),
            # sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy),
        ],
        available_actions=list(AllActions),
        # Baseline reward of 100
        get_score_method="get_num_marines_difference",
    ),
    CollectMineralsAndGas=dict(
        map_name="CollectMineralsAndGas",
        positions={
            units.Terran.CommandCenter: [(30, 36), (35, 36)],
            units.Terran.SupplyDepot:
                [(x, 31) for x in range(30, 35, 2)]
                + [(x, 41) for x in range(30, 35, 2)]
            ,
            units.Terran.Barracks: [],
        },
        multiple_positions=False,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        available_actions=list(BaseManagerActions),
        # Aim for a
        get_score_method="get_mineral_collection_rate_difference",
    ),
    CollectMinerals=dict(
        map_name="CollectMinerals",
        positions={
            units.Terran.CommandCenter: [(25, 40), (25, 30), (38, 30)],
            units.Terran.SupplyDepot:
                [(x, y) for x in range(32, 42+1, 2) for y in range(40, 46+1, 2)] # 24 supply depots
            ,
            units.Terran.Barracks: [],
        },
        multiple_positions=False,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        available_actions=list(BaseManagerActions),
        # Aim for a
        get_score_method="get_mineral_collection_rate_difference",
    ),
    BuildMarines=dict(
        map_name="BuildMarines",
        positions={
            # Seems like no extra CCs can be built here...
            # units.Terran.CommandCenter: [(36, 36), (32, 41)],
            units.Terran.CommandCenter: [(30, 36)],
            units.Terran.SupplyDepot:
                [(x, y) for x in range(29, 43+1, 2) for y in range(40, 44+1, 2)] # 24 supply depots
            ,
            units.Terran.Barracks: [(29, 29), (32, 29), (35, 29), (38, 29)]
        },
        multiple_positions=False,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        available_actions=[AllActions.NO_OP, AllActions.HARVEST_MINERALS, AllActions.RECRUIT_SCV_0, AllActions.BUILD_SUPPLY_DEPOT, AllActions.BUILD_BARRACKS, AllActions.RECRUIT_MARINE],
        # Baseline reward of 50
        get_score_method="get_num_marines_difference",
        # available_actions=list(set(list(ResourceManagerActions) + list(BaseManagerActions) + list(ArmyRecruitManagerActions)))
    ),
    BuildMarinesAlt=dict(
        map_name="BuildMarinesAlt",
        positions={
            units.Terran.CommandCenter: [(30, 36)],
            units.Terran.SupplyDepot:
                [(x, y) for x in range(29, 43+1, 2) for y in range(40, 44+1, 2)] # 24 supply depots
            ,
            units.Terran.Barracks: [(29, 29), (32, 29), (35, 29), (38, 29)]
        },
        multiple_positions=False,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        available_actions=[AllActions.NO_OP, AllActions.HARVEST_MINERALS, AllActions.RECRUIT_SCV_0, AllActions.BUILD_SUPPLY_DEPOT, AllActions.BUILD_BARRACKS, AllActions.RECRUIT_MARINE],
        # Baseline reward of 50
        get_score_method="get_num_marines_difference",
        # available_actions=list(set(list(ResourceManagerActions) + list(BaseManagerActions) + list(ArmyRecruitManagerActions)))
    ),
    DefeatRoaches=dict(
        map_name="DefeatRoaches",
        positions={
            units.Terran.CommandCenter: [],
            units.Terran.SupplyDepot: [],
            units.Terran.Barracks: [],
        },
        multiple_positions=False,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        available_actions=list(ArmyAttackManagerActions),
        get_score_method="get_reward_as_score",
    ),
    DefeatZerglingsAndBanelings=dict(
        map_name="DefeatRoaches",
        positions={
            units.Terran.CommandCenter: [],
            units.Terran.SupplyDepot: [],
            units.Terran.Barracks: [],
        },
        multiple_positions=False,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        available_actions=list(ArmyAttackManagerActions),
        reward_factor=120,
        # Baseline reward of 100
        get_score_method="get_reward_as_score",
    ),
    DefeatBase=dict(
        map_name="DefeatBase",
        positions={
            units.Terran.CommandCenter: [],
            units.Terran.SupplyDepot: [],
            units.Terran.Barracks: [],
        },
        multiple_positions=False,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        available_actions=list(ArmyAttackManagerActions),
        get_score_method="get_enemy_buildings_health",
    ),
    DefeatBaseNoPunish=dict(
        map_name="DefeatBaseNoPunish",
        positions={
            units.Terran.CommandCenter: [],
            units.Terran.SupplyDepot: [],
            units.Terran.Barracks: [],
        },
        multiple_positions=False,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        available_actions=list(ArmyAttackManagerActions),
        get_score_method="get_enemy_buildings_health",
    )
)
