import os
import collections
import numpy as np
from subprocess import Popen, PIPE

from agent import Agent
from experience import ExperienceReplay


PROCESS_NAME = "StardewModdingAPI.exe"
MODEL_PATH = "C:/Users/maxma/Desktop/real курсач/callCS/config"
DIRECTORY_PATH = "C:/Program Files (x86)/Steam/steamapps/common/Stardew Valley"

BATCH_SIZE = 500
BUFFER_SIZE = 100000
REPLAY_START_SIZE = 4000

N_FRAMES = 4      # state = 4 frames
N_ACTIONS = 9     # keys to go (+ stay)
N_FEATURES = 3072  # player, player's bullets, enemies

TRAIN = False
COMPUTER = True
SHOULD_EXECUTE = True

last_s = collections.deque(maxlen=N_FRAMES)
next_s = collections.deque(maxlen=N_FRAMES)

agent = Agent(N_FRAMES, N_FEATURES, N_ACTIONS, 1 * TRAIN, 0.999, 0.1 * TRAIN, 0.99, 1e-4, MODEL_PATH)

buffer = ExperienceReplay(BUFFER_SIZE, N_FRAMES)
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])

os.chdir(DIRECTORY_PATH)
process = Popen(PROCESS_NAME, stdin=PIPE, stdout=PIPE)

action = -1
sum_r = 0
max_t = 50
t = 0

while SHOULD_EXECUTE:
    line = process.stdout.readline().decode("cp866")[8:-2]
    process.stdout.flush()

    if line.startswith('obs'):
        state = next_s[-1]
        for i in range(96):
            for j in range(32):
                print(state[32 * i + j], end=" ")
            print()
            if (i + 1) % 32 == 0:
                print()

    if not line.startswith("data:"):
        print(line)
    else:
        line = line[6:]
        data = list(map(int, line.split(",")))

        done = data[-1]
        reward = data[-2]
        next_s.append(data[:-2])

        if TRAIN:
            if len(buffer) < REPLAY_START_SIZE:
                pass
            elif done and t % 10 == 0:
                batch = buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, is_done = batch
                agent.learn(states, actions, rewards, next_states, is_done)

            if len(last_s) == 4:
                buffer.append(Experience(np.concatenate(last_s), action, reward, np.concatenate(next_s), done))
                sum_r += reward

        last_s.append(next_s[-1])

        if len(last_s) == 4:
            action = agent.get_action(np.concatenate(last_s))
            if COMPUTER and not done:
                process.stdin.write(bytes(f'move {action}\n', encoding='cp866'))
                process.stdin.flush()
            elif not done:
                action = 0
                last_s.clear()
                next_s.clear()

        if TRAIN and done:
            t += 1
            if t % max_t == 0:
                agent.save()
                print(f"t = {t}, mean sum = {sum_r / max_t}, model saved")
                sum_r = 0
