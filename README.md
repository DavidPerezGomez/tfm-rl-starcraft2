# tfm-rl-starcraft2

Master thesis - RL - Starcraft II

## Setup

First off, this assumes that you have already setup Starcraft two, as stated in the [PySC2 instructions](https://github.com/google-deepmind/pysc2/tree/master?tab=readme-ov-file#get-starcraft-ii). The work for this project was done under Linux (Ubuntu 22.04) and used the latest [Starcraft II Linux package](https://github.com/Blizzard/s2client-proto#linux-packages) available at the time (4.10).

As for the environment, it is not very consistent between different OS, e.g. for Ubuntu 22.04 I had to go with Python 3.8 and pygame 1.9.6, for MacOS I was able to use Python 3.9 and pygame 2 (the default installed by pysc2) and for Windows I could use Python 3.10.

```bash
conda create -n tfm python=3.10
conda activate tfm
# Go to the project root
cd tfm-rl-starcraft2
# Installing the package will also install pysc2==4.0.0, pygame==1.9.6 and protobuf 3.19.6
pip install -e src/
# Install pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Install jupyterlab
pip install jupyterlab
```

You can now test running a random agent with this command:

```bash
python -m pysc2.bin.agent --map CollectMineralShards --feature_screen_size=256 --feature_minimap_size=128
```

Or you can also play with:

```bash
python -m pysc2.bin.play --map CollectMineralShards --feature_screen_size=256 --feature_minimap_size=128
```

This should write a replay, you can look for a line like this in your output:

```bash
Wrote replay to: C:/Program Files (x86)/StarCraft II\Replays\local\CollectMineralShards_2024-02-16-11-47-57.SC2Replay
```

Now, test that you can indeed show the replay:

```bash
python -m pysc2.bin.play --rgb_screen_size=1600,1200 --replay "C:/Program Files (x86)/StarCraft II\Replays\local\CollectMineralShards_2024-02-16-11-47-57.SC2Replay"
```

**Note**: At the time of this writing, we had to add this line for the replays to work on Windows, SC2 version 5.0.12:

```python
# File: pysc2/run_configs/platforms.py
# Method: get_versions
# Added at line: 102 (after the first known_versions.append and before the ret = lib.version_dict...)
    known_versions.append(
        lib.Version("latest", max(versions_found), None, None))
    # This is the extra line
    known_versions.append(
        lib.Version("5.0.12", max(versions_found), None, None))
    # End of extra line
    ret = lib.version_dict(known_versions)
```

## Maps

### List maps

```bash
python -m pysc2.bin.map_list
```

### List mini-games

```bash
python -m pysc2.bin.map_list|grep "mini_games"
```



## Agents

### Random agent

```bash
python -m pysc2.bin.agent --map CollectMineralShards --feature_screen_size=256 --feature_minimap_size=128
```

### Play as a human

```bash
python -m pysc2.bin.play --map CollectMineralShards --feature_screen_size=256 --feature_minimap_size=128
```

### Test agent

```bash
python -m pysc2.bin.agent --map CollectMineralShards --agent tfm_sc2.rl.agents.test_agent.TestAgent --use_feature_units --use_raw_units
```

```bash
python -m pysc2.bin.agent --map DefeatRoaches --agent tfm_sc2.rl.agents.test_agent.TestAgent --use_feature_units --use_raw_units
```

```bash
python -m pysc2.bin.agent --map DefeatRoaches --agent tfm_sc2.rl.agents.test_agent.TestAgent --use_feature_units --use_raw_units
```




## Troubleshooting

### `AttributeError: module 'pysc2.lib.replay' has no attribute 'get_replay_version'`

If you get an error like `AttributeError: module 'pysc2.lib.replay' has no attribute 'get_replay_version'`, then you can fix it by copying the contents of `replay.py` into the `replay/__init__.py``. Steps:

- Locate the location of the `pysc2` package in the conda environment

```bash
PYSC2_PKG_PATH=$(python -c "import pysc2; from pathlib import Path; print(Path(pysc2.__file__).parent)")
echo $PYSC2_PKG_PATH
```

- Within that folder copy the contents of `lib/replay.py` into `lib/replay/__init__.py`.

```bash
cat $PYSC2_PKG_PATH/lib/replay.py >> $PYSC2_PKG_PATH/lib/replay/__init__.py
# Show the final contents of the init file
cat $PYSC2_PKG_PATH/lib/replay/__init__.py
```

The replay should work as expected now:

```bash
python -m pysc2.bin.play --rgb_screen_size=1600,1200 --replay /home/albert/StarCraftII/Replays/RandomAgent/CollectMineralShards_2024-02-04-11-00-02.SC2Replay
```



$ python -m pysc2.bin.agent --map CollectMineralShards --agent tfm_sc2.rl.agents.test_agent.TestAgent



python -m pysc2.bin.agent --map CollectMineralShards --feature_screen_size=256 --feature_minimap_size=128

python -m pysc2.bin.play --map CollectMineralShards --rgb_screen_size=1080

python -m pysc2.bin.play -rgb_screen_size=1600,1200 --replay /home/albert/StarCraftII/Replays/RandomAgent/CollectMineralShards_2024-02-04-09-58-26.SC2Replay
