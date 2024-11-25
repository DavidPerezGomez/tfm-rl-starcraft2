
from pysc2.env import sc2_env
from pysc2.lib import features, units
from tfm_sc2.actions import AllActions, ArmyAttackManagerActions, BaseManagerActions, ArmyRecruitManagerActions

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
                # supply depots positions are very particular. sometimes rounding can throw everything off
                units.Terran.CommandCenter: [(26, 35), (56, 31), (24, 72)],
                units.Terran.SupplyDepot:
                    [(17, 31), (17, 33), (19, 28), (19, 43), (21, 26), (21, 28), (21, 43), (22, 26), (22, 40), (22, 43),
                     (24, 40), (24, 43), (26, 26), (28, 26), (28, 40), (28, 43), (30, 37), (30, 40), (30, 43), (32, 26),
                     (32, 40), (32, 43), (33, 40), (35, 26)], # 24 supply depots
                units.Terran.Barracks:
                    [(32, 28), (36, 28), (37, 38), (40, 38)],
            },
            "bottom_right": {
                units.Terran.CommandCenter: [(54, 68), (24, 72), (56, 31)],
                units.Terran.SupplyDepot:
                    [(63, 73), (63, 71), (61, 76), (61, 61), (59, 78), (59, 76), (59, 61), (57, 78), (57, 64), (57, 61),
                     (55, 64), (55, 61), (54, 78), (52, 78), (52, 64), (52, 61), (50, 67), (50, 64), (50, 61), (48, 78),
                     (48, 64), (48, 61), (46, 64), (44, 78)], # 24 supply depots,
                units.Terran.Barracks:
                    [(48, 75), (44, 75), (43, 65), (39, 65)],
            }
        },
        multiple_positions=True,
        players=[
            sc2_env.Agent(sc2_env.Race.terran),
            sc2_env.Agent(sc2_env.Race.terran),
            # sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy),
        ],
        # available_actions=list(AllActions),
        available_actions=list(BaseManagerActions) + [AllActions.BUILD_BARRACKS],
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
    CollectMineralsRandom=dict(
        map_name="CollectMineralsRandom",
        positions={
            units.Terran.CommandCenter: [(25, 40), (25, 30), (38, 30)],
            units.Terran.SupplyDepot:
                [(x, y) for x in range(32, 42+1, 2) for y in range(38, 47+1, 3)] # 24 supply depots
            ,
            units.Terran.Barracks: [],
        },
        multiple_positions=False,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        available_actions=list(BaseManagerActions),
        # Aim for a
        get_score_method="get_mineral_collection_rate_difference",
    ),
    CollectMineralsFixed=dict(
        map_name="CollectMineralsFixed",
        positions={
            units.Terran.CommandCenter: [(25, 40), (25, 30), (38, 30)],
            units.Terran.SupplyDepot:
                [(x, y) for x in range(32, 42+1, 2) for y in range(38, 47+1, 3)] # 24 supply depots
            ,
            units.Terran.Barracks: [],
        },
        multiple_positions=False,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        available_actions=list(BaseManagerActions),
        # Aim for a
        get_score_method="get_mineral_collection_rate_difference",
    ),
    BuildMarinesRandom=dict(
        map_name="BuildMarinesRandom",
        positions={
            units.Terran.CommandCenter: [(30, 36)],
            units.Terran.SupplyDepot:
                [(x, y) for x in range(28, 34+1, 2) for y in range(41, 44+1, 3)] +
                [(x, y) for x in range(36, 42+1, 2) for y in range(35, 44+1, 3)] # 24 supply depots
            ,
            units.Terran.Barracks: [(29, 29), (32, 29), (35, 29), (38, 29)]
        },
        multiple_positions=False,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        available_actions=list(ArmyRecruitManagerActions),
        get_score_method="get_army_spending_delta",
    ),
    BuildMarinesFixed=dict(
        map_name="BuildMarinesFixed",
        positions={
            units.Terran.CommandCenter: [(30, 36)],
            units.Terran.SupplyDepot:
                [(x, y) for x in range(28, 34+1, 2) for y in range(41, 44+1, 3)] +
                [(x, y) for x in range(36, 42+1, 2) for y in range(35, 44+1, 3)] # 24 supply depots
            ,
            units.Terran.Barracks: [(29, 29), (32, 29), (35, 29), (38, 29)]
        },
        multiple_positions=False,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        available_actions=list(ArmyRecruitManagerActions),
        get_score_method="get_army_spending_delta",
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
        get_score_method="get_reward_as_score",
    ),
    DefeatBases=dict(
        map_name="DefeatBases",
        positions={
            units.Terran.CommandCenter: [],
            units.Terran.SupplyDepot: [],
            units.Terran.Barracks: [],
        },
        multiple_positions=False,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        available_actions=list(ArmyAttackManagerActions),
        get_score_method="get_health_difference_score_delta",
    )
)
