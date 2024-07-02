#!/usr/bin/env python
from iiwa_msgs.srv import ConfigureControlMode, ConfigureControlModeRequest, ConfigureControlModeResponse
from iiwa_msgs.msg import ControlMode, DesiredForceControlMode, CartesianControlModeLimits, CartesianQuantity
import math
import sys
import rospy


def cartesian_quantity_from_float(value):
    quantity = CartesianQuantity()
    for attr in quantity.__slots__:
        setattr(quantity, attr, value)
    return quantity


def configure_control_mode(request):
    rospy.wait_for_service('/iiwa/configuration/ConfigureControlMode')
    print('Force mode enter')

    try:

        configure_control_mode_service = rospy.ServiceProxy('/iiwa/configuration/ConfigureControlMode', ConfigureControlMode)
        response = configure_control_mode_service(request)
        return response
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def set_position_control_mode():
    request = ConfigureControlModeRequest()
    request.control_mode = ControlMode.POSITION_CONTROL
    configure_control_mode(request)


# Use desired_force = 2N, desired_stiffness=500
def set_force_mode(cartesian_dof, desired_force, desired_stiffness, max_deviation_pos, max_deviation_rotation_in_deg):
    request = ConfigureControlModeRequest()
    request.control_mode = ControlMode.DESIRED_FORCE
    request.desired_force.cartesian_dof = cartesian_dof
    request.desired_force.desired_force = desired_force
    request.desired_force.desired_stiffness = desired_stiffness
    request.limits.max_control_force_stop = False
    request.limits.max_control_force = cartesian_quantity_from_float(-1)
    request.limits.max_cartesian_velocity = cartesian_quantity_from_float(-1)

    max_dev = request.limits.max_path_deviation
    max_dev.x = max_dev.y = max_dev.z = max_deviation_pos
    max_dev.a = max_dev.b = max_dev.c = math.radians(max_deviation_rotation_in_deg)
    configure_control_mode(request)
    print('Force mode set')

