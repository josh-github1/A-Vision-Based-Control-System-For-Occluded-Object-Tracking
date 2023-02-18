import pyrealsense2 as rs
import tensorflow as tf
import numpy as np
import cv2
import time
import math

import threading

import matplotlib.pyplot as plt
import Jetson.GPIO as GPIO

from time import sleep
from robot import Robot
from encoder import Encoder


def valueChanged(value):
    print("* New value: {}".format(value))


'''

Section for initializing values and pins for robot

'''

ro = Robot()

W = 848
H = 480

wheelRadius = 0.0408

GPIO.setmode(GPIO.TEGRA_SOC)

mode = GPIO.getmode()

pin1 = 'SPI2_CS1'

pin2 = 'SPI2_CS0'

e1 = Encoder(pin1, pin2, valueChanged)

'''

Section for the Intel Realsense pipeline

'''

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

print("[INFO] start streaming...")
pipe_profile = pipeline.start(config)
aligned_stream = rs.align(rs.stream.color)  # alignment between color and depth
point_cloud = rs.pointcloud()
print("[INFO] loading model...")
PATH_TO_CKPT = r"frozen_inference_graph2.pb"

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.compat.v1.import_graph_def(od_graph_def, name='')
    sess = tf.compat.v1.Session(graph=detection_graph)

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
# code source of tensorflow model loading: https://www.geeksforgeeks.org/ml-training-image-classifier-using-tensorflow-object-detection-api/


Rotary_counter = 0  # Start counting from 0

Current_A = 1  # Assume that rotary switch is not

Current_B = 1  # moving while we init software

LockRotary = threading.Lock()  # create lock for rotary switch

val1 = 0
val2Motor = 0
val3Motor = 0
val4Motor = 0

'''

Section for plotting data of any type

# data vs. time. Here are some of the data that you can collect:

# 1- Control command (speed or motor voltage)
# 2- Encoder data (preferably speed on all encoders)
# 3- Ball information (detection signal, as well as x-y-z positions and speeds if available) 
# 4- Filtered data (any signals that you are filtering such as the ball detection signal)

'''

voltageControl = []

speedControl = []

encoderData = []

# plotting information of ball in xyz coordinates
ball_x_coord = []
ball_y_coord = []
ball_z_coord = []  # depth
xdot_data = []
ydot_data = []
zdot_data = []

x_world = []
y_world = []
z_world = []
xdot_world = []
ydot_world = []
zdot_world = []

time_data1 = []
time_data2 = []
time_data3 = []

filteredData = []

kf_x1 = []
kf_vx1 = []
kf_y1 = []
kf_vy1 = []
kf_z1 = []
kf_vz1 = []

kf_x2 = []
kf_vx2 = []
kf_y2 = []
kf_vy2 = []
kf_z2 = []
kf_vz2 = []

ukf_x1 = []
ukf_vx1 = []
ukf_y1 = []
ukf_vy1 = []
ukf_z1 = []
ukf_vz1 = []

ukf_x2 = []
ukf_vx2 = []
ukf_y2 = []
ukf_vy2 = []
ukf_z2 = []
ukf_vz2 = []

ekf_x1 = []
ekf_vx1 = []
ekf_y1 = []
ekf_vy1 = []
ekf_z1 = []
ekf_vz1 = []

ekf_x2 = []
ekf_vx2 = []
ekf_y2 = []
ekf_vy2 = []
ekf_z2 = []
ekf_vz2 = []

'''

Section for state estimation methods

'''
import numpy as np
import sympy

from numpy import eye, array, asarray
from filterpy.kalman import KalmanFilter
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise

# deltaT = 0.25
deltaT = 0.1

################################################
############# Kalman Filter ####################
################################################
# filter to track position, velocity and acceleration.
# The sensor only reads position but in 3D space.
# The depth cam gives xyz coordinates and I can
# also derive velocity and acceleration from this.

# first testing out KF with x y position
f = KalmanFilter(dim_x=4, dim_z=2)

# assigning initial values for the states.
# these are estimated and you need to map pixel coordinates to another coordinate system
# initial values for the process state


# f.x = np.array([[1.],  # x coordinate
#                 [1.],  # y coordinate
#                 [-0.5],
#                 [-0.5]])

# another state vector config
f.x = np.array([[0],  # x position
                [0.5],  # v_z
                [0.75],  # z position
                [0.5]])  # v_z

# # defining the state transition matrix
# f.F = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])

# another state transition matrix
f.F = np.array([[1, deltaT, 0, 0], [0, 1, 0, 0], [0, 0, 1, deltaT], [0, 0, 0, 1]])

# defining the measurement function (I think this is also the observation matrix).. it's really which values can be observed
# in the state vector [x y x_dot y_dot]_transpose
f.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

# state covariance (arbitrary value.. may want to experiment with different values)
f.P *= 0.2

# also another arbitrary state covariance value
f.R = 1.307 * 10 ** (-9)

# putting white noise into the process covariance
f.Q = Q_discrete_white_noise(dim=4, dt=0.1, var=0.13)

f2 = f

################################################
############# Extended Kalman Filter ###########
################################################

'''
from math import sqrt

def HJacobian_at(x):
    """ compute Jacobian of H matrix at x """
    horiz_dist = x[0]
    altitude = x[2]
    denom = sqrt(horiz_dist**2 + altitude**2)
    return array ([[horiz_dist/denom, 0., altitude/denom]])


def hx(x):
    """ compute measurement for slant range that
    would correspond to state x.
    """
    return (x[0] ** 2 + x[2] ** 2) ** 0.5

from numpy.random import randn
import math

class RadarSim:
    """ Simulates the radar signal returns from an object
    flying at a constant altityude and velocity in 1D.
    """
    def __init__(self, dt, pos, vel, alt):
        self.pos = pos
        self.vel = vel
        self.alt = alt
        self.dt = dt
    def get_range(self):
        """ Returns slant range to the object. Call once
        for each new measurement at dt time from last call.
        """
        # add some process noise to the system
        self.vel = self.vel + .1 * randn()
        self.alt = self.alt + .1 * randn()
        self.pos = self.pos + self.vel * self.dt
        # add measurement noise
        err = self.pos * 0.05 * randn()
        slant_dist = math.sqrt(self.pos ** 2 + self.alt ** 2)
        return slant_dist + err


from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import ExtendedKalmanFilter
from numpy import eye, array, asarray
import numpy as np

deltaT = 0.05

ekf = ExtendedKalmanFilter(dim_x=4, dim_z=1)

radar = RadarSim(deltaT, pos=0., vel=100., alt=1000.)


# make an imperfect starting guess
ekf.x = array([radar.pos-100, radar.vel+100, radar.alt+1000])

ekf.F = eye(3) + array([[0, 1, 0],
                       [0, 0, 0],
                       [0, 0, 0]]) * deltaT

# ekf.x = np.array([-1., -0.5, -1., -0.5])
# ekf.F = np.array([[1, deltaT, 0, 0], [0, 1, 0, 0], [0, 0, 1, deltaT], [0, 0, 0, 1]])

range_std = 0.1
ekf.R = np.diag([range_std**2])

ekf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=deltaT, var=0.1)
ekf.Q[2,2] = 0.1
ekf.P *= 20

xs, track = [], []

import sympy

x, x_vel, y = sympy.symbols('x, x_vel y')

H = sympy.Matrix([sympy.sqrt(x**2 + y**2)])

state = sympy.Matrix([x, x_vel, y])
J = H.jacobian(state)

j1 = [J[0], J[1], J[2]]
h1 = H[0]

for i in range(int(20 / deltaT)):
    z = radar.get_range()
    track.append((radar.pos, radar.vel, radar.alt))
    ekf.update(array([z]), j1, h1)
    xs.append(ekf.x)
    ekf.predict()

xs = asarray(xs)
track = asarray(track)
time = np.arange(0, len(xs)*deltaT, deltaT)
'''


################################################
############# Unscented Kalman Filter ##########
################################################

def fx(x, dt):
    # state transition matrix
    # based on constant velocity model (x_f = v*time + x_0)
    # possibly safe to assume constant velocity model (z_f = v*time + z_0)
    F = np.array([[1, deltaT, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, deltaT],
                  [0, 0, 0, 1]],
                 dtype=float)
    return np.dot(F, x)


def hx(x):
    # measurement matrix.
    # measurements are [x_pos, z_pos]
    return np.array([x[0], x[2]])


# you actually want to keep redeclaring a UKF object
# because the deltaT changes everytime? not sure if this is good
# dt = 0.25


# creating sigma points as it is a Gaussian process
points = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=-1)

# The actual python object for the Unscented Kalman Filter
ukf = UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=deltaT, fx=fx, hx=hx, points=points)

# assuming the ball starts at the very left of the image so it has
# x-coordinate of 0 and the ball is being thrown from a distance of 0.75 meters
# initially with an arbitrarily low initial velocity of 0.5 meters/second.
ukf.x = np.array([0, 0.5, 0.75, 0.5])

# initial uncertainty
ukf.P *= 0.2

# standard deviation
z_std = 0.1

# ukf.R = np.diag([z_std ** 2, z_std ** 2])
ukf.R = np.diag([1.307 * 10 ** (-9), 1.307 * 10 ** (-9)])
ukf.Q = Q_discrete_white_noise(dim=2, dt=deltaT, var=0.01 ** 2, block_size=2)

ukf2 = ukf

set_val_kf = 0
set_val_ukf = 0
'''

Section for encoder pulses

must know how to time it in loop and must be converted into seconds
delta_t should be in seconds

clock has hz rating but have a lot of other computations.. by the time next image comes already has been x milliseconds.. have to use computer clock

also check when it starts initially.. it could start at 1000 milliseconds
let previousTime 

'''

# add delay to assist with the initial value
previousTime = time.process_time()

time.sleep(0.5)

val1_pulse = 0

encoderTickPrev = e1.getValue()

val1_pulse += 1

pulses = 0

previousL = 0

PPI = 360 / (2 * math.pi * wheelRadius)

x_axis_time = []

x_axis_time.append(previousTime)

a = []

timeSteps = []

val1_x_coord = 0
val2_x_coord = 0
val2 = 0

timeSteps = []
time1_prev = time.process_time()
sleep(0.5)

# Variables and arrays for collision avoidance logic

xCoordPasses = []
zCoordPasses = []

collisionLogicVal = 0
collisionLogicVal2 = 0
collisionLogicVal3 = 0

hitVal = 0
decrementMotorVal = 0

x_curr = 0
z_curr = 0

dxdt = 0
dzdt = 0

'''

Section for while loop - running experiment for a finite time

'''

# while encoderTickCurr < 190:
while val2 < 200 and val3Motor == 0:
    # while True:

    val2 += 1
    print("Loop iteration: {}", val2)

    time1_curr = time.process_time()
    deltaT = time1_curr - time1_prev
    timeSteps.append(deltaT)

    # I think the program goes through this while loop once and takes
    # awhile (~2-3 minutes or more ) and then it can go through the
    # while loop with very short timesteps
    frames = pipeline.wait_for_frames()
    frames = aligned_stream.process(frames)

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # also reference this github issue https://github.com/IntelRealSense/librealsense/issues/2204
    # depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    # color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    # depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
    # color_to_depth_extrin = color_frame.profile.get_extrinsics_to(depth_frame.profile)

    # depth_sensor = pipe_profile.get_device().first_depth_sensor()
    # depth_scale = depth_sensor.get_depth_scale()

    # depth_image = np.asanyarray(depth_frame.get_data())

    points = point_cloud.calculate(depth_frame)
    verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, W, 3)  # xyz

    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    scaled_size = (int(W), int(H))

    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image_expanded = np.expand_dims(color_image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                             feed_dict={image_tensor: image_expanded})

    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes).astype(np.int32)
    scores = np.squeeze(scores)

    print("[INFO] drawing bounding box on detected objects...")
    print("[INFO] each detected object has a unique color")

    # For some reason the motors only respond appropriately when outside the for loop?
    if val1 <= 0 and val2Motor == 0:
        a = 0.25
        b = 0.25
        ro.set_motors(a, b)

    print("Encoder count: {}".format(e1.getValue()))

    # 1- Control command (speed or motor voltage)
    # 2- Encoder data (preferably speed on all encoders)
    voltageControl.append(a)
    encoderData.append(e1.getValue())

    '''

    ################## pseudocode ################
    # sensors are noisy so can't rely on quadrature encoders

    # set a value to activate this intelligent decision logic

    if in a future state - ball's depth = 0 && ball x coord == 0.5 +/- 0.05
        decrement motor values until motor values = 0.

    # make predictions 5 or 10 seconds ahead
    if future distance @ current speed == near position of ball:
        decrement motor values until motor value = 0 
        # magnitude of decrement is dependent on the rate at 
        # which this collision will occur.



    '''

    # This could technically be considered an event-driven finite-state machine

    ############################# Method 1 #######################
    ############# Using the regular velocity model ###############
    if collisionLogicVal == 1:

        futureCoordinateZ = []  # local array
        futureCoordinateX = []  # local array

        stop = 10  # this is the time representing how far ahead to look into the future
        stepVal = 1  # increments of 1 second

        # Look five seconds ahead.
        for i in range(0, stop, stepVal):
            newTime = i / 2
            X_coord_f = x_curr + dxdt * newTime
            Z_coord_f = z_curr + dzdt * newTime

            futureCoordinateZ.append(Z_coord_f)
            futureCoordinateX.append(X_coord_f)

        for i in range(0, len(futureCoordinateZ)):
            if 0.40 <= futureCoordinateX[i] <= 0.6 and 0 <= futureCoordinateZ[i] <= 0.4:
                xCoordPasses.append(futureCoordinateX[i])
                zCoordPasses.append(futureCoordinateZ[i])
                hitVal += 1

        if hitVal >= len(futureCoordinateZ) - 5:
            decrementMotorVal = 1
            # so the motor won't stay stuck at
            # ro.set_motor(0.25, 0.25) from above
            val2Motor = 1

    ############################# Method 2 #######################
    ############# Using the UKF prediction #######################

    if collisionLogicVal2 == 1:

        futureCoordinateZ = []  # local array
        futureCoordinateX = []  # local array

        stop = 10  # this is the time representing how far ahead to look into the future
        stepVal = 1  # increments of 1 second

        # Look five seconds ahead.
        for i in range(0, stop, stepVal):
            newTime = i / 2
            X_coord_f = kfArr2_0 + kf_vx2 * newTime
            Z_coord_f = kfArr2_2 + kf_vy2 * newTime

            futureCoordinateZ.append(Z_coord_f)
            futureCoordinateX.append(X_coord_f)

        for i in range(0, len(futureCoordinateZ)):
            if 0.40 <= futureCoordinateX[i] <= 0.6 and 0 <= futureCoordinateZ[i] <= 0.4:
                xCoordPasses.append(futureCoordinateX[i])
                zCoordPasses.append(futureCoordinateZ[i])
                hitVal += 1

        if hitVal >= len(futureCoordinateZ) - 5:
            decrementMotorVal = 1
            val2Motor = 1  # so the motor won't stay stuck at ro.set_motor(0.25, 0.25) from above

    ############################# Method 2 #######################
    ############# Using the UKF prediction #######################

    ####################################################
    ######### machine learning predictions ############
    ####################################################
    scoreFilt = 0

    for idx in range(1):

        class_ = classes[idx]
        score = scores[idx]
        box = boxes[idx]
        # print(" [DEBUG] class: ", class_, "idx : ", idx, "num : ", num)
        print("score: ", score, "val1: ", val1)

        if a <= 0.09:
            a = 0
            val3Motor = 1

        if b <= 0.09:
            b = 0
            val3Motor = 1

        # if decrementMotorVal == 1 and val3Motor == 0:
        # a -= 0.015
        # b -= 0.015
        # ro.set_motors(a, b)

        if decrementMotorVal == 1:
            ro.stop()

        #
        # ro.stop()
        # ro.set_motors(0.25, 0.25)

        # not filtering it.. going from one timestep to another
        # there's noise in detection algorithm.. low pass filter
        # if more jitter, increase alphaFilt value
        # if less jitter, it may delay more like in the original video
        # tradeoff
        # even if box is not being drawn it will still draw the box (so it won't drop every few frames)

        # Exponentially Weighted Moving Average.
        #      Can be applied to sensors/signal data in general
        alphaFilt = 0.6
        scoreFilt = alphaFilt * scoreFilt + (1 - alphaFilt) * score

        if score <= 0.6:
            # if scoreFilt <= 0.6:

            # If the robot does not see the ball, it moves
            val1 = val1 - 1
            # time.sleep(2)

        if score > 0.6:
            # if scoreFilt > 0.5: # 1 for human
            a = time.process_time()
            time_data1.append(a)

            # If the robot does see the ball, it does not move
            # ro.stop()

            # reset the value/counter to 5
            val1 = 5

            # time.sleep(2)

            # The boxes are in [ymin xmin ymax xmax] format and always normalized relative to image size

            left = box[1] * W
            top = box[0] * H
            right = box[3] * W
            bottom = box[2] * H

            width = right - left
            height = bottom - top
            bbox = (int(left), int(top), int(width), int(height))
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

            # x,y,z of bounding box
            obj_points = verts[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])].reshape(-1, 3)
            zs = obj_points[:, 2]

            z = np.median(zs)

            ##### draw box #####
            cv2.rectangle(color_image, p1, p2, (255, 0, 0), 2, 1)

            # after calculating velocities and decided motor logic and before taking the next frame or next loop
            # velocity = (previous - current)/(currentTime - previousTime)

            '''
            # reference: https://stackoverflow.com/questions/56638290/how-do-i-accurately-retrieve-the-bounding-box-of-an-object-detected-using-tensor
            # converting from normalized values of pixels
            (im_width, im_height, _) = color_frame.shape
            xmin, ymin, xmax, ymax = box
            (xmin, xmax, ymin, ymax) = (xmin * im_width, xmax * im_width,
                                        ymin * im_height, ymax * im_height)
            '''

            # Also computing the de-projected coordinates
            # x_mid = (int(W*box[1]) + int(W*box[3])) / 2
            # y_mid = (int(H*box[0]) + int(H*box[2])) / 2
            # depth_pixel = [x_mid, y_mid]
            # depth_value = depth_image[x_mid][y_mid] * depth_scale
            # depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_value)

            #### possibly de-projection coordinates of ball? #####
            # x_ = depth_point[0]
            # y_ = depth_point[1]
            # z_ = depth_point[2]

            # Can't compute velocity on the first go of the while loop unless we have an initial x coordinate
            # of the ball to begin with.
            if val1_x_coord == 0:
                val1_x_coord += 1

                curr_time = time.process_time()
                time_data2.append(curr_time)

                ball_x_coord.append((box[1] + box[3]) / 2)
                ball_y_coord.append((box[0] + box[2]) / 2)
                ball_z_coord.append(z)

                # x_world.append(x_)
                # y_world.append(y_)
                # z_world.append(z_)

                x_prev = (box[1] + box[3]) / 2
                y_prev = (box[0] + box[2]) / 2
                z_prev = z
                prev_time = time.process_time()

            # Now that we have an initial xyz coordinate, we can now compute the velocities.
            if val2_x_coord >= 1:
                # x_world.append(x_)
                # y_world.append(y_)
                # z_world.append(z_)

                # pixel velocities
                x_curr = (box[1] + box[3]) / 2
                y_curr = (box[0] + box[2]) / 2
                z_curr = z
                curr_time = time.process_time()
                time_data2.append(curr_time)
                time_data3.append(curr_time)

                ball_x_coord.append((box[1] + box[3]) / 2)
                ball_y_coord.append((box[0] + box[2]) / 2)
                ball_z_coord.append(z)

                # computing pixel and depth velocities
                dt = curr_time - prev_time
                dxdt = (x_curr - x_prev) / dt
                dydt = (y_curr - y_prev) / dt
                dzdt = (z_curr - z_prev) / dt

                xdot_data.append(dxdt)
                ydot_data.append(dydt)
                zdot_data.append(dzdt)

                # de-projected world coordinate velocities
                # xd_curr = x_
                # yd_curr = y_
                # zd_curr = z_

                collisionLogicVal = 1

                # setting previous positions to compute velocities
                # xd_prev = xd_curr
                # yd_prev = yd_curr
                # zd_prev = zd_curr

                x_prev = x_curr
                y_prev = y_curr
                z_prev = z_curr

            val2_x_coord += 1

            '''
                    State Estimation predict and update steps
            '''
            #### pixel coordinates of the center of the object/ball #####
            x_coord = (box[1] + box[3]) / 2
            y_coord = (box[0] + box[2]) / 2
            z_coord = z

            # coordinates of the de-projected coordinates
            # and the pixel coordinates only in x and y (include c later)
            # coordArray1 = [x_, y_]
            coordArray2 = [x_coord, z_coord]

            # f.predict()
            # f.update(coordArray1)

            f2.predict()
            f2.update(coordArray2)

            # ukf.predict()
            # ukf.update(coordArray1)

            ukf2.predict()
            ukf2.update(coordArray2)

            # kfArr1 = f.x
            kfArr2 = f2.x

            '''
                kfArr1_0 = float(kfArr1[0])
                kf_x1.append(kfArr1_0)
                kfArr1_1 = float(kfArr1[1])
                kf_vx1.append(kfArr1_1)
                kfArr1_2 = float(kfArr1[2])
                kf_y1.append(kfArr1_2)
                kfArr1_3 = float(kfArr1[3])
                kf_vy1.append(kfArr1_3)
            '''

            kfArr2_0 = float(kfArr2[0])
            kf_x2.append(kfArr2_0)
            kfArr2_1 = float(kfArr2[1])
            kf_vx2.append(kfArr2_1)
            kfArr2_2 = float(kfArr2[2])
            kf_y2.append(kfArr2_2)
            kfArr2_3 = float(kfArr2[3])
            kf_vy2.append(kfArr2_3)

            # ukfArr1 = ukf.x
            ukfArr2 = ukf.x

            '''
                ukf_x1.append(ukfArr1[0])
                ukf_vx1.append(ukfArr1[1])
                ukf_y1.append(ukfArr1[2])
                ukf_vy1.append(ukfArr1[3])
            '''

            ukf_x2.append(ukfArr2[0])
            ukf_vx2.append(ukfArr2[1])
            ukf_y2.append(ukfArr2[2])
            ukf_vy2.append(ukfArr2[3])

            # ys = obj_points[:, 1]
            # ys = np.delete(ys, np.where(
            #    (zs < z - 1) | (zs > z + 1)))  # take only y for close z to prevent including background

            # my = np.amin(ys, initial=1)
            # My = np.amax(ys, initial=-1)

            # height = (My - my)  # add next to rectangle print of height using cv library
            # height = float("{:.2f}".format(height))

            print("[INFO] Object depth is: ", z, "[m]")
            # height_txt = str(z) + "[m]"
            depth_txt = str(z) + "[m]"

            # Write some Text
            font = cv2.FONT_HERSHEY_SIMPLEX

            bottomLeftCornerOfText = (p1[0], p1[1] + 20)

            fontScale = 1

            fontColor = (255, 255, 255)

            lineType = 2

            # cv2.putText(color_image, height_txt,
            #            bottomLeftCornerOfText,
            #            font,
            #            fontScale,
            #            fontColor,
            #            lineType)

            cv2.putText(color_image, depth_txt,
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)

            # 3- Ball information (detection signal, as well as x-y-z positions and speeds if available)
            #      - this would compute relative speed, not absolute speed
            #      - [ymin xmin ymax xmax] format and always normalized relative to image size
            # 4- Filtered data (any signals that you are filtering such as the ball detection signal)

            # y_prev = y_curr
            # x_prev = x_curr
            # z_prev = z_curr

    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', color_image)

    # waitKey(1) will display a frame for 1 ms, after which display will be automatically closed.
    # Since the OS has a minimum time between switching threads, the function will not wait exactly 1 ms,
    # it will wait at least 1 ms, depending on what else is running on your computer at that time.
    cv2.waitKey(1)

    # best to put after all logic and right before the next loop
    time1_prev = time1_curr

avgTimeStep = sum(timeSteps) / len(timeSteps)

print("Average time step: ")
print(avgTimeStep)

# time_data1.pop()
# time_data2.pop()


##################################################################
############ Data for plotting the ball via ######################
############ state estimation and ML        ######################
##################################################################

# Data for plotting the ball, the state estimation results
# and the distance/velocity/acceleration/voltage plots
# for the robot.

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 7

# Biased estimator of the population variance for the x-coordinate
varianceSum_x_coord = 0

mean_x_coord = sum(ball_x_coord) / len(ball_x_coord)

for i in range(0, len(ball_x_coord)):
    varianceSum_x_coord += (ball_x_coord[i] - mean_x_coord) ** 2

variance_x_coord = varianceSum_x_coord / len(ball_x_coord)

# Biased estimator of the population variance for the depth of the ball
varianceSum_z_coord = 0

mean_z_coord = sum(ball_z_coord) / len(ball_z_coord)

for i in range(0, len(ball_z_coord)):
    varianceSum_z_coord += (ball_z_coord[i] - mean_z_coord) ** 2

variance_z_coord = varianceSum_z_coord / len(ball_z_coord)

# measuring covariance matrix, R, between x and z values
covariance_sum = 0
for i in range(0, len(ball_z_coord)):
    covariance_sum = (ball_x_coord[i] - mean_x_coord) * (ball_z_coord[i] - mean_z_coord)

covariance_x_z = covariance_sum / len(ball_z_coord)

print("Variance of x")
print(variance_x_coord)

print("Variance of Z")
print(variance_z_coord)

print("Covariance between z and x")
print(covariance_x_z)

############## Scatter plot of the ball #######################
fig = plt.figure()
plt.scatter(ball_x_coord, ball_y_coord)
plt.xlabel("X-Coordinates (pixels)")
plt.ylabel("Y-Coordinates (pixels)")
plt.title(" X vs. Y Coordinates of a static ball")

fig = plt.figure()
plt.scatter(ball_x_coord, ball_z_coord)
plt.xlabel("X-Coordinates (pixels)")
plt.ylabel("Z-depth (meters)")
plt.title(" X vs. Z Location of a static ball")

#####3D Subplots w/ State Estimation Pixel coordinates#########
fig = plt.figure()

ax1 = fig.add_subplot(2, 2, 1, projection='3d')
# ax1.plot3D(ball_x_coord, ball_z_coord, ball_y_coord, 'green', label="xyz pixel/depth coordinates")
ax1.scatter(ball_x_coord, ball_z_coord, ball_y_coord, 'green', label="xyz pixel/depth coordinates")

ax1.set_xlabel('X-Coordinate (pixels)')
ax1.set_ylabel('Z-Coordinate (depth, meters)')
ax1.set_zlabel('Y-Coordinate (pixels)')
ax1.title.set_text("3D Pixel Coordinates (xyz)")
ax1.legend()

ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.plot3D(time_data2, ball_x_coord, ball_y_coord, 'green', label="xy-time")
ax2.set_xlabel('Time (seconds)')
# ax2.axes.set_xlim3d(left=0, right=120)
ax2.set_ylabel('X-Coordinate (pixels)')
ax2.set_zlabel('Y-Coordinate (pixels)')
ax2.title.set_text("XY Pixel Coordinates vs. time")
ax2.legend()

ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.plot3D(time_data2, ball_x_coord, ball_z_coord, 'green', label="xz-time")
ax3.plot3D(time_data1, kf_x2, kf_y2, 'blue', label="xz-time KF prediction")
ax3.plot3D(time_data1, ukf_x2, ukf_y2, 'red', label="xz-time UKF prediction")
ax3.set_xlabel('Time (seconds)')
# ax3.axes.set_xlim3d(left=0, right=120)
ax3.set_ylabel('X-Coordinate (pixels)')
ax3.set_zlabel('Z-Coordinate (pixels)')
ax3.title.set_text("XZ Pixel Coordinates vs. time (w/ State Estimation)")
ax3.legend()

######### 2D subplots w/ state estimation###############
fig = plt.figure()
# fig, (xy, xt, vx_t, vy_t) = plt.subplots(4)
fig, (xy, xt, vx_t) = plt.subplots(3)

xy.plot(ball_x_coord, ball_z_coord, 'green', label="xz coordinates")
xy.plot(kf_x2, kf_y2, 'blue', label="Kalman Filter (KF) prediction")
xy.plot(ukf_x2, ukf_y2, 'red', label="Unscented Kalman Filter (UKF) prediction")
xy.set_xlabel('X Coordinates (pixels)')
xy.set_ylabel('Z Coordinates (meters)')
xy.set_title('X vs. Z')
xy.legend()

xt.plot(time_data2, ball_x_coord, 'green', label="x vs. time")
xt.plot(time_data1, kf_x2, 'blue', label="KF prediction")
xt.plot(time_data1, ukf_x2, 'red', label="UKF prediction")
xt.set_xlabel('Time (seconds)')
xt.set_ylabel('X Coordinates (pixels)')
xt.set_title('X vs. Time (pixels)')
xt.legend()

kf_vx2.pop()
ukf_vx2.pop()

vx_t.plot(time_data3, xdot_data, 'green', label="v_x vs. time")
vx_t.plot(time_data3, kf_vx2, 'blue', label="KF prediction")
vx_t.plot(time_data3, ukf_vx2, 'red', label="UKF prediction")
vx_t.set_title('X-Pixel Velocity vs. Time (pixels)')
vx_t.set_xlabel('Time (seconds)')
vx_t.set_ylabel('Velocity (pixels)')
vx_t.legend()

fig = plt.figure()

plt.scatter(xCoordPasses, zCoordPasses)
plt.xlabel("X-Coordinate passes (pixels)")
plt.ylabel("Z-Coordinate passes (pixels)")

'''
######### De-Projection Coordinates ###################
######### 2D subplots w/ state estimation###############
fig = plt.figure(2)
fig, (xyd, xtd, vx_td, vy_td) = plt.subplots(4)


xyd.plot(x_world, y_world, 'green', label="xy coordinates")
xyd.plot(kf_x1, kf_y1, 'blue', label="Kalman Filter (KF) prediction")
xyd.plot(ukf_x1, ukf_y1, 'red', label="Unscented Kalman Filter (UKF) prediction")
xyd.set_xlabel('X Coordinates (de-projection)')
xyd.set_ylabel('Y Coordinates (de-projection)')
xyd.set_title('X vs. Y (de-projection)')
xyd.legend()

xtd.plot(time_data1, x_world, 'green', label="x vs. time")
xtd.plot(time_data1, kf_x1, 'blue', label="KF prediction")
xtd.plot(time_data1, ukf_x1, 'red', label="UKF prediction")
xtd.set_xlabel('Time (seconds)')
xtd.set_ylabel('X Coordinates (de-projection)')
xtd.set_title('X vs. Time (pixels)')
xtd.legend()

vx_td.plot(time_data1, xdot_data, 'green', label="v_x vs. time")
vx_td.plot(time_data1, kf_vx1, 'blue', label="KF prediction")
vx_td.plot(time_data1, ukf_vx1, 'red', label="UKF prediction")
vx_td.set_title('Velocity_x vs. time (pixels)')



#####3D Subplots w/ State Estimation Pixel coordinates#########
fig = plt.figure(3)

ax1d = fig.add_subplot(3, 2, 1, projection='3d')
ax1d.plot3D(x_world, z_world, y_world, 'green', label="xyz pixel/depth coordinates")
ax1d.set_xlabel('X-Coordinate (pixels)')
ax1d.set_ylabel('Z-Coordinate (depth, meters)')
ax1d.set_zlabel('Y-Coordinate (pixels)')
ax1d.title.set_text("3D De-Projected Coordinates (xyz)")
ax1d.legend()

ax2d = fig.add_subplot(3, 2, 2, projection='3d')
ax2d.plot3D(time_data1, x_world, y_world, 'green', label="xy-time")
ax2d.set_xlabel('Time (seconds)')
ax2d.set_ylabel('X-Coordinate (pixels)')
ax2d.set_zlabel('Y-Coordinate (pixels)')
ax2d.title.set_text("XY De-Projected Coordinates vs. time")
ax2d.legend()

ax3d = fig.add_subplot(3, 2, 3, projection='3d')
ax3d.plot3D(time_data1, x_world, y_world, 'green', label="xy-time")
ax3d.plot3D(time_data1, ukf_x1, ukf_y1, 'blue', label="xy-time KF prediction")
ax3d.plot3D(time_data1, ukf_x1, ukf_y1, 'blue', label="xy-time KF prediction")
ax3d.set_xlabel('Time (seconds)')
ax3d.set_ylabel('X-Coordinate (de-projection)')
ax3d.set_zlabel('Y-Coordinate (de-projection)')
ax3d.title.set_text("XY De-Projected Coordinates vs. time (w/ State Estimation)")
ax3d.legend()
'''

# show all figures/plots
plt.show()

print("Average time step: ", avgTimeStep)

# Stop streaming
pipeline.stop()

