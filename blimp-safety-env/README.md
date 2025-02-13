# blimp_env

## Installation

Recommended Python version is 3.10.9.

```console
git clone git@depot.ctisl.gtri.gatech.edu:trolls/task3/blimp-safety-env.git
cd blimp-safety-env
pip install -e .
```

## Usage

To run a manual control demo:

```console
# 1 --> Forward
# 2 --> Yaw left
# 3 --> Yaw right
# 4 --> Up
# 5 --> Down
python src/blimp_env/tasks/into_the_fire_basic_nav.py
```

To train an RL agent:

```console
# To see the CLI help menu
python src/blimp_env/train.py --help

# An example can be run using the following command
python src/blimp_env/train.py --config ./config/task2.yaml
```

Running and visualizing a policy from a trained agent can be accomplished using the following command (
the model weights [can be downloaded on the Box link here](https://gtri.box.com/s/k4a8ntvjtmso9xw22086ad1hqti2vj06)):

```console
unzip ./logdir/best_model.zip
python -m blimp_env.visualize --config-path config/task2.yaml
```

## License

GTRI Proprietary, Copyright (c) 2023
