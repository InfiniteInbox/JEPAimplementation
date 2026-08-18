# LeWorldModel requires action states, and slightly more complex data 
import numpy as np
# Frame 1, Draw a circle of radius 5 
def init_frame(empty_frame, x, y, r=5, w=1.5):
    h_img, w_img = empty_frame.shape
    pad = int(np.ceil(r + w))
    x_min, x_max = max(0, int(x - pad)), min(w_img, int(x + pad + 1))
    y_min, y_max = max(0, int(y - pad)), min(h_img, int(y + pad + 1))

    sub_y, sub_x = np.ogrid[y_min:y_max, x_min:x_max]
    d = np.hypot(sub_x - x, sub_y - y)
    #NEW FUNCTION CLIP! Cool :)
    alpha = np.clip((r - d) / w + 0.5, 0.0, 1.0)
    empty_frame[y_min:y_max, x_min:x_max] = np.maximum(empty_frame[y_min:y_max, x_min:x_max], alpha)
    return empty_frame
# Physics Engine 
def physics_process(x, y, init_x_vel, init_y_vel, rng, offset=2, bounce=0.9, friction=0.0, g = 9.81, hold=4, action_force=3, shape=(64, 64), r=5, w=1.5, dt=0.1, frames=60):
    h_img, w_img = shape
    curr_x, curr_y = float(x), float(y)
    vel_x, vel_y = float(init_x_vel), float(init_y_vel)

    frame_list, state_list, action_list = [], [], []
    a = 0
    b = 0

    for _ in range(frames):
        if _ % hold == 0:
            a = int(rng.integers(-2,3))
            b = int(rng.integers(-2,3))

        frame = init_frame(np.zeros(shape, dtype=np.float64), curr_x, curr_y, r=r, w=w)
        frame_list.append((frame * 255).astype(np.uint8))
        state_list.append((curr_x, curr_y, vel_x, vel_y))
        action_list.append([a,b])
        vel_x += a * action_force * dt
        vel_y += (g + (b * action_force)) * dt
        curr_x += vel_x * dt
        curr_y += vel_y * dt
        if curr_y + r > h_img - 1:
            curr_y = (h_img - 1) - r
            vel_y = -vel_y * bounce 
            vel_x *= (1.0 - friction)
        elif curr_y - r < 0:
            curr_y = r
            vel_y = -vel_y * bounce
        if curr_x + r > w_img - 1:
            curr_x = (w_img - 1) - r
            vel_x = -vel_x * bounce
        elif curr_x - r < 0:
            curr_x = r
            vel_x = -vel_x * bounce

    return np.stack(frame_list, axis=0), np.array(state_list, dtype=np.float32), np.array(action_list, dtype=np.int8)
