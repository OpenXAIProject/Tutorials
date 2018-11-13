import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk

import numpy as np
import argparse

from env import Environment 
from network import Network
from sailency import score_frame

from config import Config

FUDGE_FACTOR = 50

class Experience(object):
    def __init__(self, state, action, prediction, reward, done):
        self.state = state
        self.action = action
        self.prediction = prediction
        self.reward = reward
        self.done = done

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env', default='PongDeterministic-v0', type=str, help='gym environment')
    parser.add_argument('-m', '--mode', default='actor', type=str, help='mode of sailency')
    parser.add_argument('-d', '--density', default=5, type=int, help='density of grid of gaussian blurs')
    parser.add_argument('-r', '--radius', default=5, type=int, help='radius of gaussian blur')
    parser.add_argument('-f', '--num_frames', default=100, type=int, help='number of frames in movie')
    parser.add_argument('-i', '--first_frame', default=350, type=int, help='index of first frame')
    args = parser.parse_args()

    Config.ATARI_GAME = args.env

    env = Environment()
    network = Network("cpu:0", "network", env.get_num_actions())
    if args.env == 'PongDeterministic-v0':
        network.saver.restore(network.sess, './checkpoints/pong/network_00029000')
    elif args.env == 'BreakoutDeterministic-v0':
        network.saver.restore(network.sess, './checkpoints/breakout/network_00097000')
    else:
        raise NotImplementedError

    env.reset()
    done = False
    experiences = []

    while not done:
        # very first few frames 
        if env.current_state is None:
            env.step(0) # 0 == NOOP
            continue

        prediction, value = network.predict_p_and_v_single(env.current_state)
        action = np.argmax(prediction)
        reward, done = env.step(action)
        exp = Experience(env.previous_state, action, prediction, reward, done)
        experiences.append(exp)

    frames = []
    perturbation_maps = []
    for frame_id in range(args.first_frame, args.first_frame + args.num_frames):
        sailency = score_frame(network, experiences, frame_id, args.radius, args.density, mode=args.mode)
        pmax = sailency.max()
        
        sailency -= sailency.min() ; sailency = FUDGE_FACTOR * pmax * sailency / sailency.max()
        frames.append(experiences[frame_id].state[:, :, 3])
        perturbation_maps.append(experiences[frame_id].state[:, :, 3] + sailency)
        print(' [ %d / %d ] processing perturbation_map ... ' % (frame_id - args.first_frame, args.num_frames))

    # Visualize
    fig = plt.Figure()

    root = tk.Tk()

    label = tk.Label(root, text="Video")
    label.grid(column=0, row=0)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(column=0, row=1)

    ax_1 = fig.add_subplot(121)
    ax_2 = fig.add_subplot(122)


    def vedio(i):
        frame = frames.pop(0)
        frames.append(frame)
        ax_1.clear()
        ax_1.imshow(frame, vmin=0, vmax=1, cmap='gray')
        p_map = perturbation_maps.pop(0)
        perturbation_maps.append(p_map)
        ax_2.clear()
        ax_2.imshow(p_map, vmin=0, vmax=1, cmap='gray') #actor_sailency)

    ani = animation.FuncAnimation(fig, vedio, 1, interval=200)
    tk.mainloop()
        

