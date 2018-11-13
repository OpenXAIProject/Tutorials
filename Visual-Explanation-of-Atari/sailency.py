import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imresize

from config import Config

def occlude(img, mask):
    ret = np.zeros_like(img)
    for d in range(img.shape[2]):
        ret[:, :, d] = img[:, :, d] * (1 - mask) + gaussian_filter(img[:, :, d], sigma=3) * mask
    return ret

def get_mask(center, size, r):
    y,x = np.ogrid[-center[0]:size[0]-center[0], -center[1]:size[1]-center[1]]
    keep = x*x + y*y <= 1
    mask = np.zeros(size) ; mask[keep] = 1 # select a circle of pixels
    mask = gaussian_filter(mask, sigma=r) # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
    return mask/mask.max()

def score_frame(network, experiences, frame_id, radius, density, mode='actor'):
    # with original state
    if mode == 'actor':
        L, _ = network.predict_p_and_v_single(experiences[frame_id].state)
    elif mode == 'critic':
        _, L = network.predict_p_and_v_single(experiences[frame_id].state)
    scores = np.zeros((int(Config.IMAGE_HEIGHT / density) + 1, int(Config.IMAGE_WIDTH / density) + 1))
    for i in range(0, Config.IMAGE_HEIGHT, density):
        for j in range(0, Config.IMAGE_WIDTH, density):
            mask = get_mask(center=[i,j], size=[Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH], r=radius)
            # with occluded state
            if mode == 'actor':
                l, _ = network.predict_p_and_v_single(occlude(experiences[frame_id].state, mask))
            elif mode == 'critic':
                _, l = network.predict_p_and_v_single(occlude(experiences[frame_id].state, mask))
            scores[int(i / density), int(j / density)] = np.square(L - l).sum() * 0.5

    pmax = scores.max()
    scores = imresize(scores, size=[Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH], interp='bilinear').astype(np.float32)
    return pmax * scores / scores.max()