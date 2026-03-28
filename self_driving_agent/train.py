import torch
import sys
import argparse
import random

parser = argparse.ArgumentParser(description='Self-driving agent training')
parser.add_argument('--start', type=int, default=None, help='Spawn point index for start')
parser.add_argument('--goal', type=int, default=None, help='Spawn point index for goal')
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_actions = len(action_map)
        in_channels = 1
        episodes = 10000

        replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)
        model = DQN(num_actions, state_dim, in_channels, device)

        try:
            model.load('weights/model_ep_155')
            print("Loaded model: weights/model_ep_155")
        except:
            print("No pre-trained weights found, starting from scratch")

        # Lower exploration since we have pre-trained weights
        model.initial_eps = 0.2
        model.slope = (model.end_eps - model.initial_eps) / 250000

        run_params = env_params.copy()
        if args.start is not None:
            run_params['start_point_index'] = args.start
        if args.goal is not None:
            run_params['goal_point_index'] = args.goal

        if args.start is not None and args.goal is not None:
            print(f"Route mode: start={args.start}, goal={args.goal}")

        env = SimEnv(visuals=True, **run_params)

        for ep in range(episodes):
            if args.loop and env.start_point_index is not None:
                # Only pick goals far from start (>50m away)
                start_loc = env.spawn_points[env.start_point_index].location
                goal_candidates = [i for i in range(len(env.spawn_points))
                    if i != env.start_point_index
                    and start_loc.distance(env.spawn_points[i].location) > 50.0]
                if not goal_candidates:
                    goal_candidates = [i for i in range(len(env.spawn_points)) if i != env.start_point_index]
                env.goal_point_index = random.choice(goal_candidates)

            env.create_actors()
            env.generate_episode(model, replay_buffer, ep, action_map, eval=False)
            env.reset()

            # Save weights every 10 episodes
            if ep > 0 and ep % 10 == 0:
                model.save('weights/model_ep_{}'.format(ep))
                print(f"=== SAVED weights/model_ep_{ep} (Episode {ep}, Reward: {env.total_rewards:.0f}) ===")

            print(f"[Episode {ep} done] Reward: {env.total_rewards:.0f}")
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    finally:
        if env is not None:
            try:
                env.quit()
            except:
                pass


if __name__ == "__main__":
    run()
