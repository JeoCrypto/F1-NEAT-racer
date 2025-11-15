import math

start_pos = (422, 369)
checkpoint_0 = ((393, 376), (401, 391))

cp_mid_x = (checkpoint_0[0][0] + checkpoint_0[1][0]) // 2
cp_mid_y = (checkpoint_0[0][1] + checkpoint_0[1][1]) // 2

dx = cp_mid_x - start_pos[0]
dy = cp_mid_y - start_pos[1]

angle_rad = math.atan2(-dy, dx)
angle_deg = math.degrees(angle_rad)
pygame_angle = 90 - angle_deg

print(f"STARTING_ANGLE = {pygame_angle:.1f}")
