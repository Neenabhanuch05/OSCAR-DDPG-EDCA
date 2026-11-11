import numpy as np
import torch
import argparse
import os
#import wandb
import tqdm

from ns3gym import ns3env
from agents.our_ddpg.utils import ReplayBuffer
from agents.our_ddpg.preprocessor import Preprocessor
from agents.our_ddpg.Our_DDPG import DDPG
from agents.our_ddpg.loggers import Logger
from exceptions import AlreadyRunningException
from wrappers import EnvWrapper


# OSCAR + EDCA
# The original OSCAR DDPG agent is kept unchanged.
# Its scalar action is interpreted by cw.cc as an EDCA CW compensation value.
scenario = "basic"
agent_being_trained = "OSCAR_EDCA"
simTime = 15
stepTime = 0.01
history_length = 300
steps_per_ep = int(simTime / stepTime)
EPISODE_COUNT = 12

# Change this list for your experiment.
N_WIFI_LIST = [40, 45, 50]

for nWifi in N_WIFI_LIST:
    sim_args = {
        "simTime": simTime,
        "envStepTime": stepTime,
        "historyLength": history_length,
        "agentType": "continuous",
        "scenario": scenario,
        "nWifi": nWifi,
    }

    tags = [
        "OSCAR+EDCA",
        "DDPG",
        scenario,
        f"Station count: {nWifi}",
        f"Instances: 1",
        *[f"{key}: {sim_args[key]}" for key in list(sim_args)[:3]],
    ]

    # Use your own W&B entity/project.
    #run = wandb.init(
     #   name=f"{nWifi} OSCAR+EDCA DDPG",
      #  entity=os.environ.get("WANDB_ENTITY", "xraulz"),
       # project=os.environ.get("WANDB_PROJECT", "contention_window"),
        #tags=tags,
        #reinit=True,
    #)

    logger = Logger(False, tags, None, experiment=None)
    logger.begin_logging(EPISODE_COUNT, steps_per_ep, None, None, stepTime)
    preprocess = Preprocessor(False).preprocess

    threads_no = 1
    env = EnvWrapper(threads_no, **sim_args)
    env.reset()

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_timesteps", default=300, type=int)
    parser.add_argument("--expl_noise", default=0.1, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--discount", default=0.7, type=float)
    parser.add_argument("--tau", default=1e-3, type=float)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--load_model", default="")
    args = parser.parse_args()

    file_name = f"OSCAR_EDCA_ContentionWindow_{args.seed}"

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = 1
    action_dim = 1
    # DDPG emits [-1, 1]; EDCA extension uses [0, 896] as CWcomp.
    max_action = 1.0
    real_max_action = 896.0
    stateSize = state_dim

    policy = DDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=args.discount,
        tau=args.tau,
    )

    if args.load_model:
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = ReplayBuffer(state_dim, action_dim)

    state, done = env.reset(), False
    state = state[0][:stateSize]
    state = np.reshape(state, stateSize)
    time_step = 0

    for episode in range(EPISODE_COUNT):
        episode_reward = 0

        with tqdm.trange(1, steps_per_ep + 1) as t:
            for step in t:
                if time_step < args.start_timesteps:
                    action = np.array([[np.random.uniform(0, real_max_action)]])
                    real_action = action
                else:
                    action = (
                        policy.select_action(np.array(state))
                        + np.random.normal(
                            0, max_action * args.expl_noise, size=action_dim
                        )
                    ).clip(-max_action, max_action)

                    # DDPG output [-1,1] -> EDCA CWcomp [0,896].
                    real_action = real_max_action * (action + 1) / 2
                    real_action = np.array([real_action])

                next_state, reward, done, info = env.step(real_action)

                not_processed_state = preprocess(
                    np.reshape(next_state, (-1, len(env.envs), 1))
                )

                next_state = next_state[0][:stateSize]
                next_state = np.reshape(next_state, stateSize)

                replay_buffer.add(
                    state, action, next_state, reward, done
                )

                state = next_state
                episode_reward += reward

                if step > 300:
                    loss = {
                        "actor_loss": policy.actor_loss,
                        "critic_loss": policy.critic_loss,
                    }
                    logger.log_round(
                        state,
                        reward,
                        episode_reward,
                        info,
                        loss,
                        np.mean(not_processed_state, axis=0)[0],
                        episode * steps_per_ep + step,
                    )

                t.set_postfix(
                    mb_sent=f"{logger.sent_mb:.2f} Mb",
                    curr_speed=f"{logger.current_speed:.2f} Mbps",
                )

                if time_step >= args.start_timesteps:
                    policy.train(replay_buffer, args.batch_size)

                time_step += 1

                if done:
                    break
        logger.log_episode(
            episode_reward,
            logger.sent_mb / simTime,
            episode,
        )

        if episode < EPISODE_COUNT - 1:
            state = env.reset()
            state = state[0][:stateSize]
            state = np.reshape(state, stateSize)

    logger.end()

    if args.save_model:
        policy.save(f"./models/{file_name}")

    env.close()
    #run.finish()
