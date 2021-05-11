import os
import collections
import numpy as np
from subprocess import Popen, PIPE

from agent import Agent


PROCESS_NAME = "StardewModdingAPI.exe"
WALKING_PATH = "1.cfg"
SHOOTING_PATH = "2.cfg"
DIRECTORY_PATH = "C:/Program Files (x86)/Steam/steamapps/common/Stardew Valley"

SIZE = 32
N_FRAMES = 2      # state = 4 frames
N_ACTIONS = 9     # keys to go (+ stay)
N_FEATURES = SIZE * SIZE * 3  # player, player's bullets, enemies

COMPUTER = True
SHOULD_EXECUTE = True

last_s = collections.deque(maxlen=N_FRAMES)

walking_agent = Agent(N_FRAMES, N_FEATURES, 9, 0, 0, 0, 0.99, 1e-4, WALKING_PATH, [N_FEATURES * N_FRAMES, 2048, 1024, 512, 256])
shooting_agent = Agent(N_FRAMES, N_FEATURES, 8, 0, 0, 0, 0.99, 1e-4, SHOOTING_PATH, [N_FEATURES * N_FRAMES, 2048, 1024, 512, 256])

os.chdir(DIRECTORY_PATH)
process = Popen(PROCESS_NAME, stdin=PIPE, stdout=PIPE)

walking_action = 0
shooting_action = 0

while SHOULD_EXECUTE:
    line = process.stdout.readline().decode("cp866")[8:-2]
    process.stdout.flush()

    if line.startswith('observations') and len(last_s) > 0:
        state = last_s[-1]
        for i in range(3 * SIZE):
            for j in range(SIZE):
                print(state[SIZE * i + j], end=" ")
            print()
            if (i + 1) % SIZE == 0:
                print()
    elif line.startswith('data'):
        line = line[6:]
        data = list(map(int, line.split(",")))

        last_s.append(data)
        if len(last_s) == N_FRAMES and COMPUTER:
            walking_action = walking_agent.get_action(np.concatenate(last_s))
            shooting_action = shooting_agent.get_action(np.concatenate(last_s))
            process.stdin.write(bytes(f'move {walking_action} {shooting_action}\n', encoding='cp866'))
            process.stdin.flush()
