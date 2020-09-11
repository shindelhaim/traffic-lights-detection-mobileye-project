import numpy as np
import math


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    
    if(abs(tZ) < 10e-6):
        print('tz = ', tZ)
    
    elif (norm_prev_pts.size == 0):
        print('no prev points')
    
    elif (norm_curr_pts.size == 0):
        print('no curr points')
    
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ)
    
    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))
    
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        
        if not valid:
            Z = 0

        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    
    return corresponding_ind, np.array(pts_3D), validVec


def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point

    return (pts - pp) / float(focal)
    

def unnormalize(pts, focal, pp):
    # transform normalized pixels into pixels using the focal length and principle point

    return focal * pts + pp
    

def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    R = EM[:3, :3]
    T = EM[:3, 3]
    
    tX = T[0]
    tY = T[1]
    tZ = T[2]
    
    foe = np.array([tX / tZ, tY / tZ])

    return R, foe, tZ 


def rotate(pts, R):
    pts = np.array(pts)
    # rotate the points - pts using R
    ones = np.ones((pts.shape[0], 1))
    pts = np.hstack([pts, ones])
    temp = R @ pts.T
    rotate_pts_t = temp[:2] / temp[2]

    return (rotate_pts_t.T)[:, :2]


def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe: y = mx + n
    m = (foe[1] - p[1]) / (foe[0] - p[0])
    n = (p[1] * foe[0] - foe[1] * p[0]) / (foe[0] - p[0])

    # run over all norm_pts_rot and find the one closest to the epipolar line
    distances = []

    for i in range(norm_pts_rot.shape[0]):
        distances += [abs((m * norm_pts_rot[i][0] + n - norm_pts_rot[i][1]) / (math.sqrt(m * m + 1)))]

    distances = np.array(distances)
    index = np.argmin(distances)
    closet_point = norm_pts_rot[index]

    # return the closest point and its indexs
    
    return index, closet_point
    

def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    zX = (tZ * (foe[0] - p_rot[0])) / (p_curr[0] - p_rot[0])

    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    zY = (tZ * (foe[1] - p_rot[1])) / (p_curr[1] - p_rot[1])

    # combine the two estimations and return estimated Z
    diffX = abs(p_curr[0] - foe[0])
    diffY = abs(p_curr[1] - foe[1])

    return zX * (diffX / (diffX + diffY)) + zY * (diffY / (diffX + diffY))
