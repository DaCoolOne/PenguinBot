import math
import random

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from rlbot.utils.structures.game_data_struct import Vector3 as UI_Vec3
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState
from rlbot.utils.structures.quick_chats import QuickChats
from random import randint

from enum import Enum

boost_accel = 991.666

class State(Enum):
	SPAWN = 0
	ATTACK = 1
	HOLD = 2
	GRABBOOST = 3

class Vec3:
	def __init__(self, x=0, y=0, z=0):
		self.x = float(x)
		self.y = float(y)
		self.z = float(z)
	
	def __add__(self, val):
		return Vec3(self.x + val.x, self.y + val.y, self.z + val.z)
	
	def __sub__(self, val):
		return Vec3(self.x - val.x, self.y - val.y, self.z - val.z)
	
	def __mul__(self, val):
		return Vec3(self.x * val, self.y * val, self.z * val)
	
	def len(self):
		return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
	
	def set(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z
	
	def to_rotation(self):
		v = Vector2(self.x, self.y)
		r = math.atan2(v.y, v.x)
		r2 = math.atan(self.z / v.len())
		# r2 = math.pi * 0.3
		return Vector2(r, r2)
	
	def align_to(self, rot):
		v = Vec3(self.x, self.y, self.z)
		
		# Apply roll
		v.set(v.x, math.cos(rot.roll) * v.y + math.sin(rot.roll) * v.z, math.cos(rot.roll) * v.z - math.sin(rot.roll) * v.y)
		
		# Apply pitch
		v.set(math.cos(-rot.pitch) * v.x + math.sin(-rot.pitch) * v.z, v.y, math.cos(-rot.pitch) * v.z - math.sin(-rot.pitch) * v.x)
		
		# Apply yaw
		v.set(math.cos(-rot.yaw) * v.x + math.sin(-rot.yaw) * v.y, math.cos(-rot.yaw) * v.y - math.sin(-rot.yaw) * v.x, v.z)
		
		return v
	
	# Inverse of align_to
	def align_from(self, rot):
		v = Vec3(self.x, self.y, self.z)
		
		# Apply yaw
		v.set(math.cos(rot.yaw) * v.x + math.sin(rot.yaw) * v.y, math.cos(rot.yaw) * v.y - math.sin(rot.yaw) * v.x, v.z)
		
		# Apply pitch
		v.set(math.cos(rot.pitch) * v.x + math.sin(rot.pitch) * v.z, v.y, math.cos(rot.pitch) * v.z - math.sin(rot.pitch) * v.x)
		
		# Apply roll
		v.set(v.x, math.cos(-rot.roll) * v.y + math.sin(-rot.roll) * v.z, math.cos(-rot.roll) * v.z - math.sin(-rot.roll) * v.y)
		
		return v
	
	def UI_Vec3(self):
		return UI_Vec3(self.x, self.y, self.z)
	
	def copy(self):
		return Vec3(self.x, self.y, self.z)
	
	def flatten(self):
		return Vec3(self.x, self.y, 0.0)
	
	def normal(self, n = 1):
		l = max(self.len(), 0.0001)
		return Vec3(self.x / l * n, self.y / l * n, self.z / l * n)

def render_star(self, position: Vec3, color, size = 100):
	
	v = Vec3(1, 1, 1).normal(size)
	
	self.renderer.draw_line_3d((position - Vec3(size, 0, 0)).UI_Vec3(), (position + Vec3(size, 0, 0)).UI_Vec3(), color)
	self.renderer.draw_line_3d((position - Vec3(0, size, 0)).UI_Vec3(), (position + Vec3(0, size, 0)).UI_Vec3(), color)
	self.renderer.draw_line_3d((position - Vec3(0, 0, size)).UI_Vec3(), (position + Vec3(0, 0, size)).UI_Vec3(), color)
	
	self.renderer.draw_line_3d((position - Vec3(-v.x, v.y, v.z)).UI_Vec3(), (position + Vec3(-v.x, v.y, v.z)).UI_Vec3(), color)
	self.renderer.draw_line_3d((position - Vec3(v.x, -v.y, v.z)).UI_Vec3(), (position + Vec3(v.x, -v.y, v.z)).UI_Vec3(), color)
	self.renderer.draw_line_3d((position - Vec3(v.x, v.y, -v.z)).UI_Vec3(), (position + Vec3(v.x, v.y, -v.z)).UI_Vec3(), color)
	

def sign(n):
	if n >= 0:
		return 1
	else:
		return -1

def constrain_pi(n):
	while n > math.pi:
		n -= math.pi * 2
	while n < -math.pi:
		n += math.pi * 2
	return n

def constrain(n, mi, ma):
	return max(mi, min(n, ma))

def correct(target, val, mult = 1):
	rad = constrain_pi(target - val)
	return (rad * mult)

def within(r1, r2, n):
	return abs(constrain_pi(r1-r2)) < n

def angle_between(a, b):
	return math.acos(dot(a.normal(), b.normal()))

def Make_Vect(v):
	return Vec3(v.x, v.y, v.z)

def clamp(n, _min, _max):
	return max(_min, min(_max, n))

def dot(v1, v2):
	return v1.x*v2.x+v1.y*v2.y+v1.z*v2.z

def dot_2D(v1, v2):
	return v1.x*v2.x+v1.y*v2.y+v1.z*v2.z

def correction(car, ideal):
	v = Make_Vect(ideal).align_from(car.physics.rotation)
	return constrain_pi(math.atan2(-v.y, v.x))

def get_car_facing_vector(car):
	return Vec3(1, 0, 0).align_to(car.physics.rotation)

# Keep controls steady during flip
def Manuever_Flip(self, packet, time):
	if time > 0.4 or packet.game_cars[self.index].double_jumped:
		self.controller_state.jump = False
		self.controller_state.yaw = 0.0
		self.controller_state.pitch = 0.0
		self.controller_state.roll = 0.0
		self.controller_state.boost = False
		self.controller_state.throttle = self.flip_dir.y
	else:
		self.controller_state.jump = True
		self.controller_state.boost = False
		self.controller_state.yaw = self.flip_dir.x
		self.controller_state.pitch = -self.flip_dir.y
		self.controller_state.roll = 0.0
		self.controller_state.throttle = 1.0

# Maneuver setup
def Enter_Flip(self, packet, dir):
	self.manuever_lock = 0.5
	self.manuever = Manuever_Flip
	self.flip_dir = dir
	self.controller_state.jump = True
	self.controller_state.steer = 0.0

def Enter_High_Flip(self, packet, dir):
	self.manuever_lock = 0.6
	self.manuever = Manuever_Flip
	self.flip_dir = dir
	self.controller_state.jump = True
	self.controller_state.steer = 0.0

def Drive_To(self, packet: GameTickPacket, position, boost = False, no_overshoot = False):
	
	position = Make_Vect(position)
	
	if not self.dropshot:
		position = Vec3(constrain(position.x, -4100, 4100), constrain(position.y, -5100, 5100), position.z)
	
	if abs(position.x) < 3500 and abs(position.y) < 4500 and not self.dropshot:
		position.z = 0
	
	render_star(self, position, self.renderer.green())
	
	my_car = packet.game_cars[self.index]
	
	if my_car.has_wheel_contact:
		car_location = Make_Vect(my_car.physics.location)
		
		car_to_pos = position - car_location
		
		car_to_pos_vel = car_to_pos - Make_Vect(my_car.physics.velocity) * 0.1
		car_to_pos_vel_2 = car_to_pos - Make_Vect(my_car.physics.velocity)
		
		if no_overshoot:
			car_to_pos_vel_2 = car_to_pos + Make_Vect(my_car.physics.velocity) * 0.8
		
		steer_correction_radians = correction(my_car, car_to_pos)
		
		self.controller_state.steer = -max(-1.0, min(1.0, steer_correction_radians * 3.0))
		
		if car_to_pos.len() > 1000:
			self.controller_state.throttle = 1.0
			
			self.controller_state.boost = (abs(steer_correction_radians) < 0.5 and car_to_pos_vel_2.len() > 300.0) and boost
			
			self.controller_state.handbrake = abs(steer_correction_radians) > math.pi * 0.5
			
		else:
			self.controller_state.boost = False
			self.controller_state.handbrake = (abs(steer_correction_radians) < math.pi * 0.8) and (abs(steer_correction_radians) > math.pi * 0.3) and my_car.physics.location.z < 100.0
			self.controller_state.throttle = constrain(car_to_pos_vel_2.len() / 100, 0, 1)
			if dot_2D(car_to_pos_vel.flatten().normal(), get_car_facing_vector(my_car).flatten().normal()) < -0.7 and my_car.physics.location.z < 50.0:
				self.controller_state.throttle = -self.controller_state.throttle
				# May need to update
				self.controller_state.steer = constrain(constrain_pi(math.pi - steer_correction_radians), -1, 1)
		
		if self.controller_state.handbrake:
			self.controller_state.boost = False
		
		self.controller_state.jump = False
		self.controller_state.pitch = 0.0
	else:
		
		vel = Make_Vect(my_car.physics.velocity)
		
		car_step = Make_Vect(my_car.physics.location) + Make_Vect(my_car.physics.velocity) * 1.2
		
		# Wall landing
		if car_step.z > 150 and (abs(car_step.x) > 4096 or abs(car_step.y) > 5120) and not self.dropshot:
			if 4096 - abs(car_step.x) < 5120 - abs(car_step.y):
				Align_Car_To(self, packet, Vec3(0, 0, -1), Vec3(0, -sign(car_step.y), 0))
			else:
				Align_Car_To(self, packet, Vec3(0, 0, -1), Vec3(-sign(car_step.x), 0, 0))
		else:
			# Ground landing
			if my_car.physics.location.z > -my_car.physics.velocity.z * 1.2 and my_car.physics.location.z > 200:
				Align_Car_To(self, packet, vel.flatten().normal() - Vec3(0, 0, 2))
				self.controller_state.throttle = 1.0
				self.controller_state.boost = 0.3 > angle_between(Vec3(1, 0, 0).align_to(my_car.physics.rotation), Make_Vect(my_car.physics.velocity).flatten().normal() - Vec3(0, 0, 2))
			else:
				Align_Car_To(self, packet, vel.flatten(), Vec3(0, 0, 1))
				self.controller_state.throttle = 1.0
				self.controller_state.boost = False
		
		self.controller_state.handbrake = False
		self.controller_state.jump = False
	
	self.renderer.draw_line_3d(UI_Vec3(position.x, position.y, 100.0), my_car.physics.location, self.renderer.green())

def Collect_Boost(self, packet: GameTickPacket, position, do_boost = False, allow_flips = True, no_overshoot = False, strict = False):
	
	car = packet.game_cars[self.index]
	
	end_target = Make_Vect(position)
	
	if car.boost > 90:
		Drive_To(self, packet, position, do_boost, no_overshoot)
	else:
		target = Make_Vect(position)
		
		info = self.get_field_info()
		
		# Acts as maximum search radius
		closest_boost = 1500
		closest_boost_real = -1
		# Units per second that we're travelling
		if do_boost:
			speed = 2000.0
		else:
			speed = 1000.0
		
		max_dot = 0.2
		max_boost_dot = -0.5
		
		if strict:
			max_dot = 0.75
			max_boost_dot = 0.4
		
		car_pos = Make_Vect(car.physics.location)
		car_dir = get_car_facing_vector(car)
		
		# self.renderer.begin_rendering()
		
		# Pick boost to drive to
		for i in range(info.num_boosts):
			boost = info.boost_pads[i]
			boost_info = packet.game_boosts[i]
			car_to_boost = Make_Vect(boost.location) - car_pos
			time_to_boost = car_to_boost.len()
			d = max_dot
			
			if boost.is_full_boost: # We are allowed to turn more to get full boosts
				d = max_boost_dot
			
			car_to_boost_n = car_to_boost.normal()
			pos_to_boost_n = (Make_Vect(boost.location) - Vec3(target.x, target.y, 0.0)).normal()
			
			if -dot_2D(pos_to_boost_n, car_to_boost_n) > d and dot(get_car_facing_vector(car).normal(), car_to_boost_n) > d and boost_info.is_active: #time_to_boost > boost_info.timer:
				self.renderer.draw_line_3d(boost.location, car.physics.location, self.renderer.yellow())
				if closest_boost > time_to_boost:
					closest_boost = time_to_boost
					closest_boost_real = boost
		
		# if closest_boost_real != -1:
			# self.renderer.draw_string_3d(target.UI_Vec3(), 2, 2, str(closest_boost_real.timer), self.renderer.white())
		# self.renderer.end_rendering()
		
		p = target
		
		if closest_boost_real != -1:
			p = Make_Vect(closest_boost_real.location)
		
		self.controller_state.boost = self.controller_state.boost and do_boost
		Drive_To(self, packet, p, do_boost, no_overshoot)
		
		car_direction = get_car_facing_vector(car)
		
		steer_correction_radians = correction(car, p - car_pos)
		
		if allow_flips and abs(steer_correction_radians) < 0.1 and Make_Vect(car.physics.velocity).len() > 700 and Make_Vect(car.physics.velocity).len() < 1500 and (target - car_pos).len() > Make_Vect(car.physics.velocity).len() + 1000:
			Enter_High_Flip(self, packet, Vec3(-math.sin(steer_correction_radians), math.cos(steer_correction_radians), 0))
	

def Get_Ball_At_T(packet, prediction, time):
	delta = prediction.slices[0].game_seconds - packet.game_info.seconds_elapsed
	return prediction.slices[clamp(math.ceil(delta + time * 60), 0, len(prediction.slices) - 1)].physics

def Attack_Aim_Ball(self, packet: GameTickPacket, aim_pos: Vec3, ball_predict: Vec3):
	car = packet.game_cars[self.index]
	
	render_star(self, ball_predict, self.renderer.purple())
	
	render_star(self, aim_pos, self.renderer.green())
	
	car_rot = car.physics.rotation
	
	car_direction = get_car_facing_vector(car)
	
	a = (ball_predict - aim_pos).flatten().normal(-100)
	aim = (Vec3(a.x, a.y, 0) + car_direction * 100)
	aim_2 = Vec3(a.x, a.y, 0)
	
	aim = aim.align_from(car_rot)
	aim.y = aim.y * 0.5
	aim = aim.align_to(car_rot)
	
	aim_2 = aim_2.align_from(car_rot)
	aim_2.y = aim_2.y * 0.5
	aim_2 = aim_2.align_to(car_rot)
	
	car_to_pos = ball_predict - Make_Vect(car.physics.location) - Make_Vect(car.physics.velocity) * 0.1
	
	car_to_pos_local = car_to_pos.align_from(car.physics.rotation)
	
	car_to_ball_real = Make_Vect(packet.game_ball.physics.location) - Make_Vect(car.physics.location) - Make_Vect(car.physics.velocity) * 0.1
	
	# if ball_predict.z > 250:
		# Aerial_Hit_Ball(self, packet, aim_pos)
	if car_to_ball_real.len() < 350 and abs(car_to_ball_real.z) < 180:
		c_p = Make_Vect(packet.game_ball.physics.location) + Make_Vect(packet.game_ball.physics.velocity) * 0.15 - Make_Vect(car.physics.location) - Make_Vect(car.physics.velocity) * 0.15
		ang = correction(car, c_p)
		Enter_Flip(self, packet, Vec3(-math.sin(ang) * 2, math.cos(ang), 0.0).normal())
	elif packet.game_info.is_kickoff_pause:
		Collect_Boost(self, packet, (ball_predict - aim_2 * 2), True, False, False, True)
		self.controller_state.boost = True
	elif abs(car_to_ball_real.z) > 250:
		Collect_Boost(self, packet, (ball_predict - aim_2 * 4), True, True, True)
	else:
		Drive_To(self, packet, (ball_predict - aim_2 * (car_to_ball_real.len() * 0.005 + 0.2)), True)
	
	self.renderer.draw_line_3d(ball_predict.UI_Vec3(), (ball_predict - aim_2 * (car_to_pos.len() * 0.005 + 0.1)).UI_Vec3(), self.renderer.red())
	
	self.renderer.draw_line_3d(packet.game_ball.physics.location, aim_pos.UI_Vec3(), self.renderer.white())
	

def Get_Impulse(packet, phys, point, time):
	
	# Ensure using our type of vector
	point = Make_Vect(point)
	
	phys_pos = Make_Vect(phys.location)
	
	phys_to_ball = point - phys_pos
	
	impulse_2D = phys_to_ball.flatten()
	
	impulse_2D *= (1 / max(0.0001, time))
	
	# Worked this out a while ago
	z_vel = -(0.5 * packet.game_info.world_gravity_z * time * time - phys_to_ball.z) / max(0.0001, time)
	
	return Vec3(impulse_2D.x, impulse_2D.y, z_vel)
	

def Align_Car_To(self, packet, vector: Vec3, up = Vec3(0, 0, 0)):
	
	my_car = packet.game_cars[self.index]
	
	self.renderer.draw_line_3d(my_car.physics.location, (Make_Vect(my_car.physics.location) + vector.normal(200)).UI_Vec3(), self.renderer.red())
	
	car_rot = my_car.physics.rotation
	
	car_rot_vel = Make_Vect(my_car.physics.angular_velocity)
	
	local_euler = car_rot_vel.align_from(car_rot)
	
	align_local = vector.align_from(car_rot)
	
	local_up = up.align_from(car_rot)
	
	# Improving this
	rot_ang_const = 0.25
	stick_correct = 6.0
	
	a1 = math.atan2(align_local.y, align_local.x)
	a2 = math.atan2(align_local.z, align_local.x)
	
	if local_up.y == 0 and local_up.z == 0:
		a3 = 0.0
	else:
		a3 = math.atan2(local_up.y, local_up.z)
	
	yaw = correct(0.0, -a1 + local_euler.z * rot_ang_const, stick_correct)
	pitch = correct(0.0, -a2 - local_euler.y * rot_ang_const, stick_correct)
	roll = correct(0.0, -a3 - local_euler.x * rot_ang_const, stick_correct)
	
	max_input = max(abs(pitch), max(abs(roll), abs(yaw)))
	
	# yaw /= max_input
	# roll /= max_input
	# pitch /= max_input
	
	self.controller_state.yaw = constrain(yaw, -1, 1)
	self.controller_state.pitch = constrain(pitch, -1, 1)
	self.controller_state.roll = constrain(roll, -1, 1)
	
	self.controller_state.steer = constrain(yaw, -1, 1)
	
	self.renderer.draw_line_3d(my_car.physics.location, (Make_Vect(my_car.physics.location) + align_local.align_to(car_rot).normal(100)).UI_Vec3(), self.renderer.yellow())
	

def Aerial_To(self, packet, point, time):
	
	my_car = packet.game_cars[self.index]
	
	impulse = Get_Impulse(packet, packet.game_cars[self.index].physics, point, time) - Make_Vect(my_car.physics.velocity)
	
	impulse_2 = impulse + Vec3(0, 0, 0.2) * impulse.len()
	
	forward = Vec3(1, 0, 0).align_to(my_car.physics.rotation)
	
	if dot(impulse_2.normal(), forward) > 0.8 and my_car.physics.location.z > 200:
		Align_Car_To(self, packet, impulse_2, Make_Vect(point) - Make_Vect(my_car.physics.location))
	else:
		Align_Car_To(self, packet, impulse_2)
	
	forward = Vec3(1, 0, 0).align_to(my_car.physics.rotation)
	
	self.controller_state.boost = impulse_2.len() > max(30, angle_between(impulse_2, forward) * 800) and angle_between(impulse_2, forward) < math.pi * 0.4
	
	return impulse
	

def Maneuver_Align(self, packet, time):
	
	my_car = packet.game_cars[self.index]
	
	self.jump_timer = self.jump_timer + self.delta
	
	self.aerial_hit_time -= self.delta
	
	ball_pos = 0
	
	if self.aerial_hit_time < 0:
		ball_pos = packet.game_ball.physics.location
	else:
		ball_pos = Get_Ball_At_T(packet, self.get_ball_prediction_struct(), self.aerial_hit_time).location
	
	self.renderer.draw_line_3d(ball_pos, (Make_Vect(ball_pos) - self.aerial_hit_position).UI_Vec3(), self.renderer.green())
	
	impulse = Aerial_To(self, packet, Make_Vect(ball_pos) - self.aerial_hit_position * min(0.8+time*2, 1.5), max(self.aerial_hit_time, 0.1)) # self.aerial_hit_position, self.aerial_hit_time)
	
	local_impulse = impulse.align_from(my_car.physics.rotation)
	
	self.controller_state.jump = self.jump_timer < 0.2 and local_impulse.z > 50
	
	render_star(self, Make_Vect(Get_Ball_At_T(packet, self.get_ball_prediction_struct(), self.aerial_hit_time).location), self.renderer.red())
	
	self.renderer.draw_line_3d((Make_Vect(ball_pos) + self.aerial_hit_position * 10).UI_Vec3(), ball_pos, self.renderer.purple())
	
	if self.jump_timer > 0.3 and not my_car.double_jumped and local_impulse.z > 400:
		self.controller_state.jump = True
		self.controller_state.yaw = 0.0
		self.controller_state.pitch = 0.0
		self.controller_state.roll = 0.0
		self.controller_state.steer = 0.0
	
	if impulse.len() < 100 and self.manuever_lock > -0.2:
		self.manuever_lock = -0.2
	
	# if (impulse.len() / 900 * 40 > (my_car.boost + 10) or impulse.len() / 900 > self.aerial_hit_time) and self.manuever_lock < -0.5:
		# self.manuever_lock = 0.0
	

def Enter_Aerial(self, packet, time, aim):
	
	self.aerial_hit_time = time
	self.aerial_hit_position = aim
	
	self.jump_timer = 0.0
	
	self.has_jump = 0.6
	
	self.controller_state.jump = True
	
	self.controller_state.boost = False
	
	self.controller_state.handbrake = False
	self.controller_state.steer = 0.0
	self.controller_state.throttle = 0.02 * sign(self.controller_state.throttle)
	
	self.controller_state.pitch = 0.0
	self.controller_state.roll = 0.0
	self.controller_state.yaw = 0.0
	
	self.manuever = Maneuver_Align
	self.manuever_lock = time + 0.2
	

def Maneuver_Align_Jump(self, packet, time):
	my_car = packet.game_cars[self.index]
	
	Align_Car_To(self, packet, Make_Vect(packet.game_ball.physics.location) - Make_Vect(my_car.physics.location), Vec3(0, 0, 1))

def Enter_Align_Jump(self, packet):
	self.controller_state.jump = True
	
	self.manuever = Maneuver_Align_Jump
	self.manuever_lock = 0.5
	

def Time_to_Pos(car, position, velocity):
	car_to_pos = Make_Vect(position) - Make_Vect(car.physics.location)
	
	# Subtract 100 from the length because we contact the ball slightly sooner than we reach the point
	len = max(0, car_to_pos.len() - 200)
	vel = Make_Vect(velocity)
	v_len = vel.len() * dot(car_to_pos.normal(), vel.normal())
	
	# curve:
	# f(t) = 0.5 * boost_accel * t ^ 2 + velocity * t
	
	# Analysis of t:
	# Solve for t when f(t) = len
	# Zeros of t: let c = len
	# 0.5 * boost_accel * t ^ 2 + velocity * t - len = 0
	
	# t = ( -velocity + sqrt(velocity^2 - 4(boost_accel)(-len)) ) / ( 2 * (boost_accel) )
	
	accel_time = (-v_len + math.sqrt(v_len * v_len + 4 * boost_accel * len)) / (2 * boost_accel)
	
	# However, the car speed maxes out at 2300 uu, so we need to account for that by stopping acceleration at 2300 uu. To do this we
	# calculate when we hit 2300 uu and cancel out any acceleration that happens after that
	
	# f(t) = 0.5 * boost_accel * t ^ 2 + velocity * t
	# Derivative:
	# v(t) = boost_accel * t + velocity
	# Solve for t when v(t) = 2300
	# 2300 = boost_accel * t + velocity
	# 2300 - velocity = boost_accel * t
	# ( 2300 - velocity ) / boost_accel = t
	
	max_vel_time = (2300 - v_len) / boost_accel
	
	a = 0
	
	if max_vel_time < accel_time:
		
		# plug time into position function
		pos = 0.5 * boost_accel * max_vel_time * max_vel_time + v_len * max_vel_time
		
		# Calculate additional distance that needs to be traveled
		extra_vel = len - pos
		
		# Add additional time onto velocity
		a = max_vel_time + extra_vel / 2300
		
	else:
		a = accel_time
	
	return a
	

def Aerial_Hit_Ball(self, packet: GameTickPacket, target: Vec3):
	
	my_car = packet.game_cars[self.index]
	
	time = 0
	ball_pos = packet.game_ball.physics.location
	car_pos = Make_Vect(my_car.physics.location) # + Make_Vect(my_car.physics.velocity) * 0.25
	
	aim = Vec3()
	
	up_vect = Vec3(0, 0, 200)
	
	# Car gets instantaneous velocity increase of 300 uu from first jump, plus some for second jump
	if not my_car.double_jumped:
		up_vect = Vec3(50, 0, 300).align_to(my_car.physics.rotation)
	
	for i in range(20):
		
		ball_pos = Make_Vect(Get_Ball_At_T(packet, self.get_ball_prediction_struct(), time).location)
		
		aim = (target - ball_pos).normal(50)
		
		time = Time_to_Pos(my_car, ball_pos - aim, Make_Vect(my_car.physics.velocity) + up_vect) * 1.3
		
	
	self.controller_state.handbrake = False
	
	if ball_pos.z < 250:
		Attack_Aim_Ball(self, packet, target, ball_pos)
	elif time > 2:
		Collect_Boost(self, packet, ball_pos - aim.normal(3000), True, True)
	else:
		
		impulse_raw = Get_Impulse(packet, my_car.physics, ball_pos, time)
		
		impulse = impulse_raw - Make_Vect(my_car.physics.velocity) - up_vect
		
		impulse_n = impulse_raw.normal() - (Make_Vect(my_car.physics.velocity) + up_vect).normal()
		
		impulse_local = impulse_n.align_from(my_car.physics.rotation)
		
		forward_vect = Vec3(1, 0, 0).align_to(my_car.physics.rotation)
		
		if my_car.has_wheel_contact:
			
			a = correction(my_car, ball_pos - aim)
			
			Drive_To(self, packet, ball_pos - aim)
			
			# if abs(a) < 0.1: # or impulse_local.flatten().len() < 100:
				# self.controller_state.throttle = constrain(impulse_local.x * 3, -1, 1)
				# self.controller_state.steer = constrain(a, -1, 1)
			
			# if dot(my_car.physics.velocity, get_car_facing_vector(my_car)) < 0.0:
				# self.controller_state.steer = constrain(-self.controller_state.steer, -1, 1)
			
			# if abs(dot(Vec3(1, 0, 0), impulse_local.flatten().normal())) < 0.2 or impulse_local.flatten().len() < impulse_local.z * 0.5:
				# self.controller_state.handbrake = True
				# self.controller_state.throttle = 1.0
			
			# self.controller_state.boost = self.controller_state.throttle abs(a) < 0.2 and impulse_local.x > 3
			
			self.renderer.draw_line_3d(my_car.physics.location, (car_pos + impulse_raw).UI_Vec3(), self.renderer.red())
			self.renderer.draw_line_3d(my_car.physics.location, (car_pos + Make_Vect(my_car.physics.velocity) + up_vect).UI_Vec3(), self.renderer.yellow())
			
			render_star(self, ball_pos, self.renderer.yellow())
		else:
			Drive_To(self, packet, ball_pos)
		
		# Drive_To(self, packet, ball_pos, True)
		
		if dot(impulse_raw.normal(), (Make_Vect(my_car.physics.velocity) + up_vect).normal()) > 0.95 and impulse.len() / 900 * 40 < (my_car.boost + 10) and impulse.len() / 900 < time:
			self.jump_timer += self.delta
		else:
			self.jump_timer = 0.0
		
		# self.controller_state.jump = my_car.has_wheel_contact and self.jump_timer > 0.2
		
		# ((car_pos - Make_Vect(packet.game_ball.physics.location)).len() < 300 and not my_car.has_wheel_contact)
		if self.jump_timer > 0.15: # impulse_local.flatten().len() < max(75, time * 400)
			Enter_Aerial(self, packet, time, aim)
		

def Approximate_Time_To_Ball(prediction, car_index, packet, resolution, acceleration = 0, boost = True):
	
	car = packet.game_cars[car_index]
	
	car_pos = Make_Vect(car.physics.location)
	car_vel = Make_Vect(car.physics.velocity)
	
	arieal_speed = max(1000, car_vel.len()) + acceleration
	
	time_to_reach_ball = 0
	refine = 0
	
	slice = 0
	ball_pos = Make_Vect(prediction.slices[0].physics.location)
	car_to_ball = ball_pos - car_pos
	
	for i in range(0, len(prediction.slices) - 1, resolution):
		slice = clamp(math.ceil(time_to_reach_ball * 60), 0, len(prediction.slices) - 5)
		
		ball_pos = Make_Vect(prediction.slices[slice].physics.location)
		
		car_to_ball = ball_pos - car_pos
		
		time_to_reach_ball = (car_to_ball.len() - 100) / arieal_speed
		
		if time_to_reach_ball < prediction.slices[slice].game_seconds:
			return time_to_reach_ball
		
		# refine = refine + 1
	
	# while ball_pos.z > 250 and slice < len(prediction.slices) - 5:
		# slice += 1
		# ball_pos = Make_Vect(prediction.slices[slice].physics.location)
	
	# car_to_ball = ball_pos - car_pos
	
	# time_to_reach_ball = car_to_ball.len() / arieal_speed
	
	if boost:
		return time_to_reach_ball * (2.0 - car.boost * 0.01)
	else:
		return time_to_reach_ball

def Grab_Boost_Pad(self, packet, target):
	
	big_pads = []
	
	pos = target
	
	info = self.get_field_info()
	
	for index in range(info.num_boosts):
		boost = info.boost_pads[index]
		if boost.is_full_boost and packet.game_boosts[index].is_active:
			big_pads.append(boost)
	
	if len(big_pads) > 0:
		l = 4000
		p = Make_Vect(packet.game_cars[self.index].physics.location)
		for boost in big_pads:
			l2 = (Make_Vect(boost.location) - p).len()
			if l2 < l:
				l = l2
				pos = Make_Vect(boost.location)
	
	Drive_To(self, packet, pos, True)
	

class Hit_Detector:
	
	def __init__(self):
		self.prediction_pos = Vec3(0, 0, 0)
		
	
	def search_prediction_for(self, prediction, position, min_slice):
		for i in range(min_slice, min_slice + 30):
			if (position - Make_Vect(prediction.slices[i].physics.location)).len() < 5:
				return True
		return False
	
	def step(self, prediction):
		
		changed = not self.search_prediction_for(prediction, self.prediction_pos, 15)
		
		self.prediction_pos = Make_Vect(prediction.slices[20].physics.location)
		
		return changed
		
	

class Plan:
	
	def __init__(self, agent):
		self.state = State.SPAWN
		self.agent = agent
		self.timer = 0.0
		self.hit_detect = Hit_Detector()
		self.def_pos_1 = Vec3()
		self.def_pos_2 = Vec3()
		self.def_pos = Vec3()
		
		self.attacking_car = 0
		
		self.attack_car_dist = 0
		self.p_attack_car_dist = 0
		
		self.aim_pos = Vec3()
		
	
	def get_team_cars(self, packet):
		team_cars = []
		
		team = packet.game_cars[self.agent.index].team
		
		for i in range(packet.num_cars):
			car_to_ball = Make_Vect(packet.game_ball.physics.location) - Make_Vect(packet.game_cars[i].physics.location)
			if team == packet.game_cars[i].team and self.agent.index != i and not packet.game_cars[i].is_demolished:
				team_cars.append(i)
		
		return team_cars
	
	# def get_messages():
		# for i in range(100):  # process at most 100 messages per tick.
			# try:
				# msg = self.agent.matchcomms.incoming_broadcast.get_nowait()
			# except Empty:
				# break
		
	
	def eval(self, agent, packet: GameTickPacket, own_goal):
		
		# print("Evaluation", packet.game_info.seconds_elapsed)
		# print("Frame 1", agent.get_ball_prediction_struct().slices[0].game_seconds)
		
		own_goal = Make_Vect(own_goal)
		
		team = self.get_team_cars(packet)
		
		prediction = agent.get_ball_prediction_struct()
		
		dist = 1000
		closest_team_mate = -1
		car_behind_ball = False
		for index in team:
			d = Approximate_Time_To_Ball(prediction, index, packet, 5, 0, not packet.game_info.is_kickoff_pause)
			
			car_pos = Make_Vect(packet.game_cars[index].physics.location)
			
			if index == self.attacking_car:
				d *= 3
			
			if dot(car_pos - own_goal, Make_Vect(Get_Ball_At_T(packet, prediction, d).location) - car_pos) < 0.0:
				d *= 5
			else:
				car_behind_ball = True
			
			if not packet.game_cars[index].has_wheel_contact:
				d *= 5
			
			if dist > d:
				dist = d
				closest_team_mate = index
		
		time_to_ball = Approximate_Time_To_Ball(agent.get_ball_prediction_struct(), agent.index, packet, 5, 0, not packet.game_info.is_kickoff_pause)
		
		car_pos = Make_Vect(packet.game_cars[agent.index].physics.location)
		is_behind_ball = dot(car_pos - own_goal, Make_Vect(Get_Ball_At_T(packet, prediction, time_to_ball).location) - car_pos) > 0.0
		
		if not is_behind_ball:
			time_to_ball *= 5.0
		
		if not packet.game_cars[agent.index].has_wheel_contact:
			time_to_ball *= 5.0
		
		if time_to_ball <= dist and (is_behind_ball or not car_behind_ball):
			self.attacking_car = agent.index
			self.state = State.ATTACK
		else:
			self.attacking_car = closest_team_mate
			if packet.game_info.is_kickoff_pause:
				self.state = State.GRABBOOST
			else:
				self.state = State.HOLD
		
		self.attack_car_to_ball = (Make_Vect(packet.game_ball.physics.location) - Make_Vect(packet.game_cars[self.attacking_car].physics.location))
		self.attack_car_vel = Make_Vect(Make_Vect(packet.game_cars[self.attacking_car].physics.velocity))
		
	
	def update(self, agent, packet: GameTickPacket):
		
		info = agent.get_field_info()
		my_car = packet.game_cars[agent.index]
		
		my_goal = 0
		opponent_goal = 0
		
		for index in range(info.num_goals):
			goal = info.goals[index]
			if goal.team_num == my_car.team:
				my_goal = goal
			else:
				opponent_goal = goal
		
		self.attack_car_to_ball = (Make_Vect(packet.game_cars[self.attacking_car].physics.location) - Make_Vect(packet.game_ball.physics.location))
		self.attack_car_vel = Make_Vect(packet.game_cars[self.attacking_car].physics.velocity)
		
		has_hit_ball = self.hit_detect.step(agent.get_ball_prediction_struct())
		
		# print(packet.game_info.is_kickoff_pause)
		
		if self.state == State.SPAWN or has_hit_ball or dot(my_goal.direction, self.attack_car_to_ball) < 0.0 or ((dot(self.attack_car_vel, self.attack_car_to_ball) < 0.0 or self.attack_car_vel.len() < 200) and self.attack_car_to_ball.len() > 250) or packet.game_cars[self.attacking_car].is_demolished:
			if self.state != State.GRABBOOST or packet.game_cars[agent.index].boost > 20:
				self.eval(agent, packet, my_goal.location)
		
		# Dropshot
		if agent.dropshot:
			
			dir_y = sign(my_goal.location.y)
			
			self.def_pos_1 = Vec3(sign(packet.game_ball.physics.location.x) * -1000, packet.game_ball.physics.location.y + dir_y * 1000, 0.0)
			self.def_pos_2 = Make_Vect(packet.game_ball.physics.location) + Vec3(0, dir_y * 2000, 0) * 3500
			if sign(dir_y) == sign(packet.game_ball.physics.location.y):
				self.aim_pos = Make_Vect(packet.game_ball.physics.location) + Vec3(0, dir_y * 1000, 1000)
			else:
				self.aim_pos = Make_Vect(packet.game_ball.physics.location) + Vec3(0, dir_y * 1000, -1000)
			self.aggro = True
			
		# 3v3
		else:
			
			ball = Get_Ball_At_T(packet, agent.get_ball_prediction_struct(), 3)
			a = ball.location.y - my_goal.location.y
			b = ball.location.y - opponent_goal.location.y
			
			# Defensive positioning
			if abs(a) < 3000 or (abs(b) > 4000 and (ball.velocity.y + sign(my_goal.direction.y) * 1000) * sign(my_goal.direction.y) < 0.0):
				self.def_pos_2 = Vec3(sign(packet.game_ball.physics.location.x) * -3500, my_goal.location.y + my_goal.direction.y * 500, 0.0)
				self.def_pos_1 = Make_Vect(my_goal.location).flatten() - Vec3(sign(packet.game_ball.physics.location.x) * 500, 0, 0)
				self.aim_pos = Vec3(sign(packet.game_ball.physics.location.x) * 3600, packet.game_ball.physics.location.y + my_goal.direction.y * 1000, 500.0)
				self.aggro = False
			# Offensive positioning
			else:
				self.def_pos_1 = Vec3(sign(packet.game_ball.physics.location.x) * -1000, packet.game_ball.physics.location.y + opponent_goal.direction.y * 1000, 0.0)
				b_p = Make_Vect(packet.game_ball.physics.location)
				b_p.x *= 0.5
				self.def_pos_2 = b_p + Make_Vect(opponent_goal.direction) * 4000
				self.aim_pos = Make_Vect(opponent_goal.location) + Vec3(0, 0, -400)
				self.aggro = True
		
		team = self.get_team_cars(packet)
		
		if len(team) < 1: 
			self.def_pos = self.def_pos_1
		else:
			other_car_index = 0
			
			if self.attacking_car == team[0]:
				other_car_index = 1
			
			if len(team) <= other_car_index:
				self.def_pos = self.def_pos_1
			else:
				
				c1 = packet.game_cars[agent.index].physics.location
				c2 = packet.game_cars[team[other_car_index]].physics.location
				
				l1 = (self.def_pos_1 - c1).len() * (2 - packet.game_cars[agent.index].boost * 0.01) # + (self.def_pos_2 - c2).len()
				l2 = (self.def_pos_1 - c2).len() * (2 - packet.game_cars[team[other_car_index]].boost * 0.01) # + (self.def_pos_2 - c1).len()
				
				if l1 < l2:
					self.def_pos = self.def_pos_1
				else:
					self.def_pos = self.def_pos_2
				
			
		
		
		# render_star(agent, self.def_pos_1, agent.renderer.blue())
		# render_star(agent, self.def_pos_2, agent.renderer.blue())
		
		# render_star(agent, Make_Vect(packet.game_cars[self.attacking_car].physics.location), agent.renderer.red())
		
		# print(dot(self.attack_car_vel, self.attack_car_to_ball))
		
		# if self.attacking_car == agent.index:
			# render_star(agent, Make_Vect(packet.game_cars[agent.index].physics.location), agent.renderer.red())
		# else:
			# render_star(agent, Make_Vect(packet.game_cars[agent.index].physics.location), agent.renderer.blue())
		
		# if len(team) > 0:
			# agent.renderer.draw_line_3d(packet.game_cars[agent.index].physics.location, packet.game_cars[team[0]].physics.location, agent.renderer.purple())
		
		# if len(team) > 1:
			# agent.renderer.draw_line_3d(packet.game_cars[agent.index].physics.location, packet.game_cars[team[1]].physics.location, agent.renderer.purple())
	

class PenguinBot(BaseAgent):
	
	def get_goal(self, team):
		for i in range(self.get_field_info().num_goals):
			goal = self.get_field_info().goals[i]
			if goal.team_num == team:
				return goal
	
	def initialize_agent(self):
		self.controller_state = SimpleControllerState()
		
		self.delta = 0
		self.prev_time = 0
		
		self.plan = Plan(self)
		
		self.manuever = Maneuver_Align
		self.manuever_lock = 0.0
		
		self.flip_dir = Vec3()
		
		self.jump_timer = 0.0
		
	
	def brain(self, packet: GameTickPacket):
		
		my_goal = self.get_goal(packet.game_cars[self.index].team)
		other_goal = self.get_goal((packet.game_cars[self.index].team + 1) % 2)
		
		self.plan.update(self, packet)
		
		if packet.game_cars[self.index].is_demolished:
			self.plan.state = State.SPAWN
			return
		
		if self.plan.state == State.ATTACK or ((Make_Vect(packet.game_cars[self.index].physics.location) - Make_Vect(my_goal.location)).len() < 2000 and dot(Make_Vect(packet.game_cars[self.index].physics.location) - Make_Vect(packet.game_ball.physics.location), Make_Vect(my_goal.location) - Make_Vect(packet.game_ball.physics.location)) > 0.0):
			Aerial_Hit_Ball(self, packet, self.plan.aim_pos)
		elif self.plan.state == State.GRABBOOST:
			Grab_Boost_Pad(self, packet, self.plan.def_pos)
		else:
			#if packet.game_cars[self.index].boost > 50:
			Collect_Boost(self, packet, self.plan.def_pos, not self.dropshot, True, True, False)
			# else:
				# Grab_Boost_Pad(self, packet)
	
	def get_output(self, packet: GameTickPacket):
		
		self.dropshot = packet.num_tiles > 0
		
		self.renderer.begin_rendering()
		
		time = packet.game_info.seconds_elapsed
		self.delta = time - self.prev_time
		self.prev_time = time
		
		if self.manuever_lock <= 0.0:
			self.brain(packet)
		else:
			self.manuever_lock = self.manuever_lock - self.delta
			self.manuever(self, packet, self.manuever_lock)
		
		self.renderer.end_rendering()
		
		return self.controller_state
	










