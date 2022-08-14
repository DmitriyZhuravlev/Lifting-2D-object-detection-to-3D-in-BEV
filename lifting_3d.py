import cv2
import numpy as np
import traceback
import math
from collections import deque
from enum import Enum

# Vehicle size  893.8270396238155 407.2407150568322


VEHICLE_L = 3571
VEHICLE_W = 1627

# 3800 (мм) до 6100 (мм), ширина - от 1500 (мм) до 2500 (мм),

vehicle_l = None
vehicle_w = None

IMAGE_H = 40840
IMAGE_W = 8160

pixels_in_mm = None

debug = False
max_debug_index = 300
min_debug_index = 200

thikness = 20


class cv_colors(Enum):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    PURPLE = (247, 44, 200)
    ORANGE = (44, 162, 247)
    MINT = (239, 255, 66)
    YELLOW = (2, 255, 250)


class b3d:
    def __init__(self, cls, center):
        self.cls = cls
        self.center = center


persp_mat = []
inv_mat = []

pts = [deque(maxlen=30) for _ in range(9999)]

def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)

def warp_perspective(p, matrix):
    px = (matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2]) / (
        (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2])
    )
    py = (matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2]) / (
        (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2])
    )
    p_after = (int(px), int(py))  # after transformation
    return p_after


def warp(a, i, matrix):
    return warp_perspective(a[i], matrix)


def to_warp(a, matrix):
    if a is not None and len(a) > 0:
        return list(map(lambda i: warp(a, i, matrix), range(0, len(a))))
    else:
        return None


def iv(a):
    return (a[0], -a[1])


def to_iv(array):
    if array is None:
        return None
    converted = []
    for a in array:
        converted.append(iv(a))

    return converted


def get_intersect(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1, a2, b1, b2])  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    if z == 0:  # lines are parallel
        return (float("inf"), float("inf"))
    return (x / z, y / z)


def get_angle(a, b, c):
    ang = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    return ang + math.pi * 2 if ang < 0 else ang


def get_frame(warp_corners, orient, h, w, bev, mov_angle, color):
    """
    a2  - a3
    |     |
    a1    a4

    nearest a1
    """
    a = to_iv(warp_corners)

    x2_x1 = a[1][0] - a[0][0]
    y2_y1 = a[1][1] - a[0][1]
    x4_x3 = a[3][0] - a[2][0]
    y4_y3 = a[3][1] - a[2][1]

    A = [[np.sin(orient), -np.cos(orient)], [y2_y1, -x2_x1]]
    v = [
        a[3][0] * np.sin(orient) - a[3][1] * np.cos(orient),
        a[0][0] * y2_y1 - a[0][1] * x2_x1,
    ]

    k0, k1 = np.linalg.solve(A, v)
    k = (k0, k1)

    if k1 < a[0][1] or k1 > a[1][1]:
        return None, None

    # if debug: print("K: ", k)

    l = math.dist(a[3], k)

    c0 = ((l - h) * a[0][0] + h * a[3][0]) / l
    c1 = ((l - h) * a[0][1] + h * a[3][1]) / l

    if c0 < a[0][0] or c0 > a[3][0]:
        return None, None

    c = (c0, c1)


    A = [[np.sin(orient), -np.cos(orient)], [y2_y1, -x2_x1]]
    v = [
        c[0] * np.sin(orient) - c[1] * np.cos(orient),
        a[0][0] * y2_y1 - a[0][1] * x2_x1,
    ]

    b0, b1 = np.linalg.solve(A, v)
    b = (b0, b1)

    if b1 < a[0][1] or b1 > a[1][1]:
        return None, None


    A = [[np.cos(orient), np.sin(orient)], [y4_y3, -x4_x3]]
    v = [
        c[0] * np.cos(orient) + c[1] * np.sin(orient),
        a[2][0] * y4_y3 - a[2][1] * x4_x3,
    ]

    d0, d1 = np.linalg.solve(A, v)
    d = (d0, d1)

    if d1 < a[3][1] or d1 > a[2][1]:
        return None, None


    error = abs(math.dist(c, d) - h)
    if debug: print("Side error :", error)

    center = ((b[0] + d[0]) / 2, (b[1] + d[1]) / 2)
    f = (c[0] + (center[0] - c[0]) * 2, c[1] + (center[1] - c[1]) * 2)

    #w_points = [iv(c), iv(b), iv(f), iv(d)] #, iv(center)]
    w_points = to_iv([c, b, f, d])

    if debug:
        int_center = (int(center[0]), - int(center[1]))
        int_w_points = np.array(w_points, dtype = np.int32)
        for i in range(4):
            cv2.line(bev, int_w_points[i], int_w_points[(i + 1) % 4], color, thikness)
        if mov_angle is not None:
            bev = cv2.arrowedLine(bev, int_center, (int(int_center[0] + h * math.cos(mov_angle)), int(int_center[1] - h * math.sin(mov_angle))), color, thikness)


    return error, w_points




def get_orientation(bev, ps_bev, mat, inv_mat, dim_w, dim_h):
    """
    a2  - a3
    |     |
    a1    a4

    nearest a1
    """


    B = iv(ps_bev[3])
    D = iv(ps_bev[2])
    K = iv(ps_bev[1])
    A = iv(ps_bev[0])


    d = math.dist(A, B)
    a = dim_h  # math.dist( ps_bev[2], ps_bev[3] )
    b = dim_w  # math.dist( ps_bev[2], ps_bev[1] )

    alpha = get_angle(B, A, K)
    beta = get_angle(D, B, A)

    e = math.sin(alpha) * (a * math.sin(beta) + b * math.cos(beta))
    f = math.sin(beta) * (a * math.cos(alpha) + b * math.sin(alpha))
    g = d * math.sin(alpha) * math.sin(beta)

   # cos_theta = e / (math.sqrt(e * e + f * f))
    #sin_theta = f / (math.sqrt(e * e + f * f))

    theta = math.atan(f / e)

    try:
        psi = math.acos(g / (math.sqrt(e * e + f * f)))

        if debug: print("psi :", psi)
        if psi >= math.pi / 2 or psi < 0:
            if debug: print("Error in PSI")

        phi1 = theta + psi
        phi2 = theta - psi  # + 2* math.pi

        b1ba = get_angle((B[0] + 1, B[1]), B, A)
        # if debug: print("AB orientation : ", math.degrees(b1ba))

        g_phi_1 = -phi1 + b1ba  # + math.pi  # - math.pi
        g_phi_2 = -phi2 + b1ba  # - math.pi  # - math.pi

        # return [phi1 + ba0, -phi2 + ba0]
        return [g_phi_1, g_phi_2]

    except:
        if debug: print("Error acos domain")
        return None


def get_bottom(bev, ps_bev, mat, inv_mat, dim_w, dim_h, mov_angle, color):

    orientation = get_orientation(bev, ps_bev, mat, inv_mat, dim_w, dim_h)

    best_corners = None
    best_error = 1e09
    best_orientation = None
    best_angle_error = math.pi

    if orientation is None:
        return None, None, None

    for g_phi in orientation:
        # mov_angle = None
        if mov_angle is not None:
            angle_error1 = abs(g_phi - mov_angle) % (2 * math.pi)
            angle_error2 = abs(g_phi + math.pi - mov_angle) % (2 * math.pi)
            angle_error = min(angle_error1, angle_error2)
            # g_phi = mov_angle
        error, corners = get_frame(ps_bev, g_phi, dim_h, dim_w, bev, mov_angle, color)

        # TODO check error
        if corners is not None and (
            mov_angle is not None
            and best_angle_error > angle_error
            or mov_angle is None
            and best_error > error
        ):
            if best_error < error:
                if debug: print("WARNING: TODO check angle and size errors")
            best_error = error
            best_corners = corners
            if mov_angle is not None:
                best_angle_error = angle_error
            best_orientation = g_phi

    return best_error, best_orientation, best_corners


def get_obj_size(cls):
    # ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck'

    # 3800 (мм) до 6100 (мм), ширина - от 1500 (мм) до 2500 (мм),
    match cls:
        case 2:
            # Длина автомобиля составляет от 4200 до 4500 мм для хэтчбеков и от 4500 до 4700 мм для седанов и лифтбеков, ширина — 1,6 — 1,75 м.
            #if debug: print("car")
            return (pixels_in_mm * 4300, pixels_in_mm * 1600)
        case 3:
            # 2100х780х1100 мм.
            return (pixels_in_mm * 2100, pixels_in_mm * 780)
        case 7:
            return (pixels_in_mm * 12000, pixels_in_mm * 2500)
        case _:
            return None
            # raise ValueError("Class not supported")

# TODO refactor to numpy vectors operations
def get_bottom_variants(id, bev, box_2d, mat, inv_mat, cls):  # dim_w, dim_l):

    mov_angle = get_motion_direction(id, box_2d, cls, mat)

    # left top right bottom
    box_corners = [i for i in box_2d]

    xmin = box_corners[0]
    ymin = box_corners[1]
    xmax = box_corners[2]
    ymax = box_corners[3]

    ps = np.array([[xmin, ymax], [xmin, ymin], [xmax, ymin], [xmax, ymax]], dtype=np.float32)

    ps_bev = to_warp(list(ps), mat)

    if debug:
        int_ps_bev = np.array(ps_bev, dtype = np.int32)
        for i in range(4):
            cv2.line(bev, int_ps_bev[i], int_ps_bev[(i + 1) % 4], cv_colors.YELLOW.value, thikness)

  
    dim_l, dim_w = get_obj_size(cls)
    if dim_l is None or dim_w is None:
        return None, None, None

    best_error, best_orient, best_corners = get_bottom(bev, ps_bev, mat, inv_mat, dim_w, dim_l, mov_angle, cv_colors.MINT.value)
    if mov_angle is not None:
        error, orient, corners = get_bottom(
            bev, ps_bev, mat, inv_mat, dim_l, dim_w, mov_angle + math.pi / 2, cv_colors.PURPLE.value
        )
    else:
        error, orient, corners = get_bottom(
            bev, ps_bev, mat, inv_mat, dim_l, dim_w, None, cv_colors.PURPLE.value
        )

    if error is not None and (
        (best_error is not None and best_error > error) or best_error is None
    ):
        best_corners = corners

    untop_corners = None

    if debug: print(best_corners)
    if best_corners is not None:
        untop_corners = to_warp(best_corners, inv_mat)
        h = untop_corners[2][1] - box_2d[1]
    else:
        h = None
    # corners = [get_bottom( bev, box_2d, mat, inv_mat, 562.8294240451921, 236.0 )] #, get_bottom( bev, box_2d, mat, inv_mat, 236.0, 562.8294240451921 )]
    
    if best_corners is None:
        return None, None, None
    
    lower_face = np.array(untop_corners, dtype = np.float32)
    upper_face = get_upper_face(box_2d, lower_face)
    rectangle = np.array(best_corners, dtype = np.float32)
    
    return lower_face, upper_face, rectangle


def get_motion_direction(id, box_2d, cls, persp_mat):
    x1, y1, x2, y2 = [int(i) for i in box_2d]
    center = (int(((x1) + (x2)) / 2), int(((y1) + (y2)) / 2))
    pts[id].append(b3d(cls, center))

    if len(pts[id]) > 1:
        t2 = t1 = warp_perspective(pts[id][-2].center, persp_mat)
        t1 = warp_perspective(pts[id][-1].center, persp_mat)

        v_x = t2[0] - t1[0]
        v_y = t2[1] - t1[1]  # inverse y
        angle = math.atan2(v_y, v_x)

        return angle

    return None


def hconcat_resize(img_list, interpolation=cv2.INTER_CUBIC):
    # take minimum hights
    h_min = min(img.shape[0] for img in img_list)

    # image resizing
    im_list_resize = [
        cv2.resize(
            img, (int(img.shape[1] * h_min / img.shape[0]), h_min), interpolation=interpolation
        )
        for img in img_list
    ]

    # return final image
    return cv2.hconcat(im_list_resize)


def get_mat():
    x_offset = 3000
    y_offset = 30000

    src = np.array([[1683, 1480], [1576, 1428], [1976, 1414], [2090, 1466]], dtype=np.float32)

    k = VEHICLE_L / VEHICLE_W
    w = math.dist(src[0], src[3])
    h = k * w

    global pixels_in_mm
    pixels_in_mm = w / VEHICLE_W
    # 3800 (мм) до 6100 (мм), ширина - от 1500 (мм) до 2500 (мм),
    # vehicle_l = 5000 * pixels_in_mm
    # vehicle_w = 2000 * pixels_in_mm
    # vehicle_l = VEHICLE_L * pixels_in_mm
    # vehicle_w = VEHICLE_W * pixels_in_mm
    if debug: print("Mean vehicle size:  ", vehicle_l, vehicle_w)
    if debug: print("Test vehicle w:  ", w)

    dst = np.array(
        [
            [0 + x_offset, 0 + h + y_offset],
            [0 + x_offset, 0 + y_offset],
            [w + x_offset, 0 + y_offset],
            [w + x_offset, h + y_offset],
        ],
        dtype=np.float32,
    )

    return cv2.getPerspectiveTransform(src, dst), cv2.getPerspectiveTransform(dst, src)


def transform(img, mat):
    dim = (IMAGE_W, IMAGE_H)
    return cv2.warpPerspective(
        img, mat, dim, flags=cv2.INTER_CUBIC , borderMode = cv2.BORDER_REPLICATE)


def hconcat_resize(img_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(img.shape[0] for img in img_list)

    im_list_resize = [
        cv2.resize(
            img, (int(img.shape[1] * h_min / img.shape[0]), h_min), interpolation=interpolation
        )
        for img in img_list
    ]

    # return final image
    return cv2.hconcat(im_list_resize)

def get_upper_face(box_2d, lower_face):
    upper_face = np.full([4, 2], None, dtype=np.float32)

    xmin = box_2d[0]
    ymin = box_2d[1]
    xmax = box_2d[2]
    ymax = box_2d[3]

    #E
    #h = lower_face[2, 1] - box_2d[1]
    upper_face[2] = lower_face[2] - np.array([0, lower_face[2, 1] - box_2d[1]], dtype=np.float32)
    #F
    with np.errstate(all = "raise"):
        try:
            right_van = get_intersect(lower_face[1], lower_face[2], lower_face[0], lower_face[3])
            upper_face[1] = get_intersect(upper_face[2], right_van, (xmin, ymin), (xmin, ymax))
            #G
            left_van = get_intersect(lower_face[2], lower_face[3], lower_face[0], lower_face[1])
            upper_face[3] = get_intersect(upper_face[2], left_van, (xmax, ymin), (xmax, ymax))
            upper_face[0] = get_intersect(left_van, upper_face[1], right_van, upper_face[3])
        except:
            if debug: print("lines are parallel")
            k1 = math.dist(lower_face[0], lower_face[1]) / math.dist(lower_face[2], lower_face[3])
            upper_face[1] = lower_face[1] - np.array([0, k1 * (lower_face[2, 1] - box_2d[1])], dtype=np.float32)
            k2 = math.dist(lower_face[0], lower_face[3]) / math.dist(lower_face[2], lower_face[1])
            upper_face[3] = lower_face[3] - np.array([0, k2 * (lower_face[2, 1] - box_2d[1])], dtype=np.float32)
            upper_face[0] = lower_face[0] - np.array([0, k1 * k2 * (lower_face[2, 1] - box_2d[1])], dtype=np.float32)
        
    return upper_face


def is_crop(img_h, img_w, box_2d):
    #return False
    # TODO update e
    e = 30

    xmin = box_2d[0]
    ymin = box_2d[1]
    xmax = box_2d[2]
    ymax = box_2d[3]

    if xmin < e or ymin < e or xmax > img_w - e or ymax > img_h - e:
        return True

    return False


def crop(img):
    h = 5000
    w = 1000

    w0 = 500
    return img[h:IMAGE_H, w0 : IMAGE_W - w]
