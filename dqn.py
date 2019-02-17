import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
from model import MyModel
from memory import DataMemory

import torch

import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer

import argparse
import random
import numpy as np
from collections import deque

ACTION_NUM = 2
IMG_HEIGHT, IMG_WIDTH = 80, 80
GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3000. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-6

def initalize(game_state):
    do_nothing = np.zeros(ACTION_NUM)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)

    x_t = process_img(x_t)

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)

    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])

    return s_t

def process_img(x_t):
    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t, (IMG_HEIGHT, IMG_WIDTH))
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))

    x_t = (x_t / 255.0).astype(np.float32)

    return x_t

def get_next_state(s_t, x_t):
    x_t = x_t.reshape(1, 1, x_t.shape[0], x_t.shape[1])

    s_n = np.append(x_t, s_t[:, :3, :, :], axis=1)

    return s_n

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    parser.add_argument('--checkpoint', default="final/easy.pth", type=str, metavar='PATH',
            help="path to your checkpoint")
    parser.add_argument('--resume', default="", type=str, metavar='PATH',
            help="path to resume(defailt: none)")
    args = vars(parser.parse_args())

    model = MyModel()
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    loss_func = torch.nn.MSELoss()
    game_state = game.GameState()

    memory = DataMemory(REPLAY_MEMORY)

    s_t = initalize(game_state)
    
    if args['mode'] == 'Run':
        OBSERVE = 999999999
        epsilon = FINAL_EPSILON
        print ("Loading weight from {}...".format(args["checkpoint"]))
        if torch.cuda.is_available():
            checkpoint = torch.load(args["checkpoint"])
        else:
            checkpoint = torch.load(args["checkpoint"], map_location='cpu')
        model.load_state_dict(checkpoint["state_dict"])
        print ("Weight loaded successfully")
        model.eval()
    else:
        if args["resume"]:
            if torch.cuda.is_available():
                checkpoint = torch.load(args["resume"])
            else:
                checkpoint = torch.load(args["resume"], map_location='cpu')
            model.load_state_dict(checkpoint["state_dict"])
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON
        model.train()

    t = 0
    while (True):
        loss_to_show = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])
        #choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:
            if (random.random() <= epsilon):
                print("----------Random Action----------")
                if random.random() > 0.1:
                    action_index = 0
                else:
                    action_index = 1
                # action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                if torch.cuda.is_available():
                    q = model(torch.from_numpy(s_t).cuda())
                    # print (q.cpu().detach().numpy())
                    max_Q = np.argmax(q.cpu().detach().numpy())
                else:
                    q = model(torch.from_numpy(s_t))
                    # print (q.detach().numpy())
                    max_Q = np.argmax(q.detach().numpy())
                action_index = max_Q
                a_t[max_Q] = 1

        #We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #run the selected action and observed next state and reward
        x_t_colored, r_t, terminal = game_state.frame_step(a_t)

        x_t = process_img(x_t_colored)

        s_n = get_next_state(s_t, x_t)

        # store the transition in DataMemory
        memory.add(s_t, action_index, r_t, s_n, float(terminal))

        #only train if done observing
        if t > OBSERVE:
            state_t, action_t, reward_t, state_next, terminal = \
                memory.gen_minibatch(BATCH)

            if torch.cuda.is_available():
                Q_output = model(torch.from_numpy(state_t).cuda())
                Q_eval = Q_output[range(Q_output.shape[0]), action_t].view(BATCH, 1)

                Q_next = model(torch.from_numpy(state_next).cuda()).detach()
                Q_next_mask = Q_next.max(1)[0].view(BATCH, 1)
                Q_target = torch.from_numpy(reward_t).cuda() + \
                    GAMMA * Q_next_mask * torch.from_numpy(terminal).cuda()

            else:
                Q_output = model(torch.from_numpy(state_t))
                Q_eval = Q_output[range(Q_output.shape[0]), action_t].view(BATCH, 1)

                Q_next = model(torch.from_numpy(state_next)).detach()
                Q_next_mask = Q_next.max(1)[0].view(BATCH, 1)
                Q_target = torch.from_numpy(reward_t) + \
                    GAMMA * Q_next_mask * torch.from_numpy(terminal)

            loss = loss_func(Q_eval, Q_target)

            loss_to_show += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # save progress every 10000 iterations
        if (args["mode"] == "Train") and (t % 10000 == 0):
            print("Saving checkpoint...")
            torch.save({
                'iters': t,
                'state_dict': model.state_dict(),
                }, 'easy/model_{}.pth'.format(t))
        t = t + 1
        s_t = s_n

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "| STATE", state, \
            "| EPSILON", epsilon, "| ACTION", action_index, "| REWARD", r_t, \
            "| Loss ", loss_to_show)

    print("Episode finished!")

if __name__ == "__main__":
    main()




