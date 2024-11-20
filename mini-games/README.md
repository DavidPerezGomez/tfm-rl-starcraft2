To be able to run these mini-games with PySC2 they need to be added to the StarCraft II maps list, as indicated in the [PySC2 documentation](https://github.com/google-deepmind/pysc2?tab=readme-ov-file#get-the-maps), by copying the map files to the mini-games folder.

`/path/to/StarCraftII/Maps/mini_games/`

Additionally, the maps need to be configured in the PySC2 library by including their names in the file `pysc2/maps/mini_games.py`.

```python
mini_games = [
    "BuildMarines",  # 900s
    "BuildMarinesAlt",  # 900s
    "CollectMinerals",  # 900s
    "CollectMineralsAndGas",  # 420s
    "CollectMineralShards",  # 120s
    "DefeatRoaches",  # 120s
    "DefeatZerglingsAndBanelings",  # 120s
    "DefeatBase", # 180s
    "DefeatBaseNoPunish", # 180s
    "FindAndDefeatZerglings",  # 180s
    "MoveToBeacon",  # 120s
]
```

Said file is part of PySC2, and not this project, so the changes will not be added to version control and will need to be reconfigured if the library is updated or reinstalled.