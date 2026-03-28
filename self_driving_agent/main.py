import os
import sys
import argparse
import random

parser = argparse.ArgumentParser(description='Self-driving agent evaluation')
parser.add_argument('scenario', nargs='?', default='ClearNoon', help='Weather scenario')
parser.add_argument('--start', type=int, default=None, help='Spawn point index for start')
parser.add_argument('--goal', type=int, default=None, help='Spawn point index for goal')
parser.add_argument('--once', action='store_true', help='Stop after first episode')
parser.add_argument('--loop', action='store_true', help='Pick new random goal after reaching one')
args = parser.parse_args()

from DQN_Control.replay_buffer import ReplayBuffer
from DQN_Control.model import DQN

from config import action_map, env_params
from utils import *
from environment import SimEnv


def run():
    env = None
    try:
        buffer_size = 1e4
        batch_size = 32
        state_dim = (128, 128)
        device = "cpu"
        num_actions = len(action_map)
        in_channels = 1
        episodes = 10000

        replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)
        model = DQN(num_actions, state_dim, in_channels, device)

        model.load('weights/model_ep_155')
        print("Loaded model: weights/model_ep_155")

        run_params = env_params.copy()
        run_params['max_iter'] = 2000000
        if args.start is not None:
            run_params['start_point_index'] = args.start
        if args.goal is not None:
            run_params['goal_point_index'] = args.goal

        if args.start is not None and args.goal is not None:
            print(f"Route mode: start={args.start}, goal={args.goal}")

        env = SimEnv(visuals=True, **run_params)

        for ep in range(episodes):
            # If looping, pick a new random goal each episode (keep same start)
            if args.loop and env.start_point_index is not None:
                goal_candidates = [i for i in range(len(env.spawn_points)) if i != env.start_point_index]
                env.goal_point_index = random.choice(goal_candidates)
                print(f"Loop mode: new goal = {env.goal_point_index}")

            env.create_actors()
            result = env.generate_episode(model, replay_buffer, ep, action_map, eval=True)
            env.reset()
            if result is False:
                break
            if args.once:
                print("Done (--once flag)")
                break
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        if env is not None:
            try:
                env.reset()
                env.quit()
            except:
                pass


if __name__ == "__main__":
    run()
