import numpy as np
import torch
import gym
import argparse
import os
import wandb
import tqdm
import subprocess
import time

class AlreadyRunningException(Exception):
    pass

from ns3gym import ns3env
from ns3gym.start_sim import find_waf_path


class EnvWrapper:
    def __init__(self, no_threads, **params):
        self.params = params
        self.no_threads = no_threads
        self.ports = [13968+i+np.random.randint(40000) for i in range(no_threads)]
        self.commands = self._craft_commands(params)

        self.SCRIPT_RUNNING = False
        self.envs = []

        self.run()

        for port in self.ports:
            env = ns3env.Ns3Env(
                port=port,
                stepTime=params['envStepTime'],
                startSim=0,
                simSeed=0,
                simArgs=params,
                debug=False
            )
            self.envs.append(env)

        self.SCRIPT_RUNNING = True

    def run(self):
        if self.SCRIPT_RUNNING:
            raise AlreadyRunningException("Script is already running")

        for cmd, port in zip(self.commands, self.ports):
            subprocess.Popen(['bash', '-c', cmd])
        self.SCRIPT_RUNNING = True

    def _craft_commands(self, params):
        try:
            waf_pwd = find_waf_path("./")
        except FileNotFoundError:
            import sys
            sys.path.append("../../")
            waf_pwd = find_waf_path("../../")

        command = f'{waf_pwd} --run "OSCAR_EDCA_DDPG'
        for key, val in params.items():
            command += f" --{key}={val}"

        commands = []
        for p in self.ports:
            commands.append(command + f' --openGymPort={p}"')

        return commands

    def reset(self):
        if not hasattr(self, "_first_reset_done"):
            self._first_reset_done = True
        else:
            for env in self.envs:
                env.close()

            self.envs = []
            self.SCRIPT_RUNNING = False
            self.run()
            time.sleep(2)

            for port in self.ports:
                env = ns3env.Ns3Env(
                    port=port,
                    stepTime=self.params['envStepTime'],
                    startSim=0,
                    simSeed=0,
                    simArgs=self.params,
                    debug=False
                )
                self.envs.append(env)

        obs = []
        for env in self.envs:
            obs.append(env.reset())

        return obs

    def step(self, actions):
        next_obs, reward, done, info = [], [], [], []

        for i, env in enumerate(self.envs):
            no, rew, dn, inf = env.step(actions[i].tolist())
            next_obs.append(no)
            reward.append(rew)
            done.append(dn)
            info.append(inf)

        return np.array(next_obs), np.array(reward), np.array(done), np.array(info)

    def close(self):
        time.sleep(5)
        for env in self.envs:
            env.close()
        self.SCRIPT_RUNNING = False
