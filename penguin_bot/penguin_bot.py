import math

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from rlbot.utils.structures.game_data_struct import Vector3 as UI_Vec3
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState
from rlbot.utils.structures.quick_chats import QuickChats
from random import randint

from enum import Enum

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
		
		# Apply yaw
		v.set(math.cos(-rot.yaw) * v.x + math.sin(-rot.yaw) * v.y, math.cos(-rot.yaw) * v.y - math.sin(-rot.yaw) * v.x, v.z)
		
		# Apply pitch
		v.set(v.x, math.cos(rot.pitch) * v.y + math.sin(rot.pitch) * v.z, math.cos(rot.pitch) * v.z - math.sin(rot.pitch) * v.y)
		
		# Apply roll
		v.set(math.cos(rot.pitch) * v.x + math.sin(rot.pitch) * v.z, v.y, math.cos(rot.pitch) * v.z - math.sin(rot.pitch) * v.x)
		
		return v
	
	def UI_Vec3(self):
		return UI_Vec3(self.x, self.y, self.z)
	
	def copy(self):
		return Vec3(self.x, self.y, self.z)
	
	def normal(self, n = 1):
		l = self.len()
		return Vec3(self.x / l * n, self.y / l * n, self.z / l * n)


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
	return min(1, max(-1, constrain_pi(target - val) * mult))

def within(r1, r2, n):
	return abs(constrain_pi(r1-r2)) < n

def angle_between(a, b):
	return math.cos(dot(a, b) / (a.len() * b.len()))

def Make_Vect(v):
	return Vec3(v.x, v.y, v.z)

def Rotation_Vector(rot):
	pitch = float(rot.pitch)
	yaw = float(rot.yaw)
	facing_x = math.cos(pitch) * math.cos(yaw)
	facing_y = math.cos(pitch) * math.sin(yaw)
	facing_z = math.sin(pitch)
	return Vec3(facing_x, facing_y, facing_z)

def clamp(n, _min, _max):
	return max(_min, min(_max, n))

def dot_2D(v1, v2):
	return v1.x*v2.x+v1.y*v2.y

def dot(v1, v2):
	return v1.x*v2.x+v1.y*v2.y+v1.z*v2.z

# Manuevers (Anything that requires very specific input and cannot be interupted)

# Half Flip = back flip + forward pitch as hard as possible then air roll
def Manuever_Half_Flip(self, packet, time):
	if time > 1.4:
		self.controller_state.jump = False
		self.controller_state.yaw = 0.0
		self.controller_state.pitch = 0.0
		self.controller_state.roll = 0.0
		self.controller_state.boost = False
		self.controller_state.throttle = 1.0
	elif time > 1.3 and not packet.game_cars[self.index].double_jumped:
		self.controller_state.jump = True
		self.controller_state.boost = False
		self.controller_state.yaw = 0.0
		self.controller_state.pitch = 1.0
		self.controller_state.roll = 0.0
		self.controller_state.throttle = 1.0
	elif time > 1.1:
		self.controller_state.jump = False
		self.controller_state.boost = False
		self.controller_state.yaw = 0.0
		self.controller_state.pitch = 0.0
		self.controller_state.roll = 0.0
		self.controller_state.throttle = 1.0
	elif time > 0.7:
		self.controller_state.jump = False
		self.controller_state.boost = time < 0.9
		self.controller_state.yaw = 0.0
		self.controller_state.pitch = -1.0
		self.controller_state.roll = 0.0
		self.controller_state.throttle = 1.0
	else:
		self.controller_state.roll = correct(0.0, packet.game_cars[self.index].physics.rotation.roll)
		self.controller_state.jump = False
		self.controller_state.throttle = 1.0
		self.controller_state.boost = True
		self.controller_state.yaw = 0.0
		self.controller_state.pitch = 0.0

# Keep controls steady during flip
def Manuever_Flip(self, packet, time):
	if time > 0.9 or packet.game_cars[self.index].double_jumped:
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
	self.manuever_lock = 1.0
	self.manuever = Manuever_Flip
	self.flip_dir = dir
	self.controller_state.jump = True

def Enter_Half_Flip(self, packet):
	self.manuever_lock = 1.7
	self.manuever = Manuever_Half_Flip
	self.controller_state.jump = True
	self.controller_state.pitch = 1.0
	self.controller_state.boost = False

def Drive_To(self, packet: GameTickPacket, position):
	
	my_car = packet.game_cars[self.index]
	
	if my_car.has_wheel_contact:
		car_location = Vector2(my_car.physics.location.x, my_car.physics.location.y)
		
		car_to_pos = Vector2(position.x, position.y) - car_location
		
		car_direction = get_car_facing_vector(my_car)
		
		steer_correction_radians = car_direction.correction_to(car_to_pos)
		
		self.controller_state.throttle = 1.0
		
		self.controller_state.steer = -max(-1.0, min(1.0, steer_correction_radians * 2.0))
		
		self.controller_state.boost = abs(steer_correction_radians) < 0.2
		
		self.controller_state.handbrake = abs(steer_correction_radians) > math.pi * 0.5 and Make_Vect(my_car.physics.velocity).len() > 200
		
		self.controller_state.jump = False
		self.controller_state.pitch = 0.0
	else:
		car_location = Vector2(my_car.physics.location.x, my_car.physics.location.y)
		
		car_direction = get_car_facing_vector(my_car)
		
		steer_correction_radians = car_direction.correction_to(my_car.physics.velocity)
		
		car_rot = my_car.physics.rotation
		
		self.controller_state.throttle = 1.0
		
		self.controller_state.roll = correct(0.0, car_rot.roll)
		self.controller_state.yaw = -max(-1.0, min(1.0, steer_correction_radians * 2.0))
		
		self.controller_state.pitch = correct(0.0, car_rot.pitch)
		
		self.controller_state.steer = -max(-1.0, min(1.0, steer_correction_radians * 2.0))
		
		self.controller_state.handbrake = False
		self.controller_state.boost = False
		self.controller_state.jump = False

# Variant of Drive_To() that does not use boost
def Flip_Drive_To(self, packet: GameTickPacket, position):
	
	my_car = packet.game_cars[self.index]
	
	if my_car.has_wheel_contact:
		car_location = Vector2(my_car.physics.location.x, my_car.physics.location.y)
		
		car_to_pos = Vector2(position.x, position.y) - car_location
		
		car_direction = get_car_facing_vector(my_car)
		
		steer_correction_radians = car_direction.correction_to(car_to_pos)
		
		if abs(steer_correction_radians) < 0.3:
			self.flip_wait_timer = self.flip_wait_timer + self.delta
		else:
			self.flip_wait_timer = 0.0
		
		self.controller_state.throttle = 1.0
		
		self.controller_state.steer = -max(-1.0, min(1.0, steer_correction_radians * 2.0))
		
		self.controller_state.handbrake = abs(steer_correction_radians) > math.pi * 0.5 and Make_Vect(my_car.physics.velocity).len() > 200
		
		self.controller_state.boost = False
		self.controller_state.jump = False
		self.controller_state.pitch = 0.0
		
		if self.flip_wait_timer > 0.5:
			Enter_Flip(self, packet, Vector2(0, 1.0))
			self.flip_wait_timer = 0.0
		
	else:
		car_location = Vector2(my_car.physics.location.x, my_car.physics.location.y)
		
		car_direction = get_car_facing_vector(my_car)
		
		steer_correction_radians = car_direction.correction_to(my_car.physics.velocity)
		
		car_rot = my_car.physics.rotation
		
		self.controller_state.throttle = 1.0
		
		self.controller_state.roll = correct(0.0, car_rot.roll)
		self.controller_state.yaw = -max(-1.0, min(1.0, steer_correction_radians * 2.0))
		self.controller_state.pitch = correct(0.0, car_rot.pitch)
		
		self.controller_state.steer = -max(-1.0, min(1.0, steer_correction_radians * 2.0))
		
		self.controller_state.handbrake = False
		self.controller_state.boost = False
		self.controller_state.jump = False

def Approximate_Time_To_Ball(prediction, car_index, packet, resolution, acceleration = 0, boost = True):
	
	car = packet.game_cars[car_index]
	
	ball_pos = Make_Vect(packet.game_ball.physics.location)
	car_pos = Make_Vect(car.physics.location)
	car_vel = Make_Vect(car.physics.velocity)
	
	arieal_speed = max(1500, car_vel.len()) + acceleration
	
	time_to_reach_ball = 0
	refine = 0
	
	car_to_ball = Make_Vect(prediction.slices[0].physics.location) - car_pos
	
	while refine < resolution:
		car_to_ball = Make_Vect(prediction.slices[clamp(math.ceil(time_to_reach_ball * 60), 0, len(prediction.slices) - 1)].physics.location) - car_pos
		
		target_vel_xy2 = Vector2(car_to_ball.x, car_to_ball.y)
		
		time_to_reach_ball = car_to_ball.len() / arieal_speed
		
		target_vel_xy = target_vel_xy2 * (arieal_speed / target_vel_xy2.len())
		
		z_vel = (car_to_ball.z - 0.5 * packet.game_info.world_gravity_z * time_to_reach_ball * time_to_reach_ball) / time_to_reach_ball
		
		# velocity we want to be going
		target_vel = Vec3(target_vel_xy.x, target_vel_xy.y, z_vel)
		
		refine = refine + 1
	
	if boost:
		return time_to_reach_ball * (2 - car.boost * 0.01)
	else:
		return time_to_reach_ball

def Get_Ball_At_T(prediction, time_to_reach_ball):
	return prediction.slices[clamp(math.ceil(time_to_reach_ball * 60), 0, len(prediction.slices) - 1)].physics

def Hit_Ball_To(self, packet: GameTickPacket, aim_pos: Vec3):
	exit_condition = True
	
	info = self.get_field_info()
	
	own_goal = 0
	for goal in info.goals:
		if goal.team_num == packet.game_cars[self.index].team:
			own_goal = goal
			break
	
	if self.is_arieal:
		
		self.jump_timer += self.delta
		
		if self.flip_timer > 0.0:
			self.flip_timer -= self.delta
		else:
			prediction = self.get_ball_prediction_struct()
			
			car = packet.game_cars[self.index]
			
			ball_pos = Make_Vect(packet.game_ball.physics.location)
			car_pos = Make_Vect(car.physics.location)
			car_vel = Make_Vect(car.physics.velocity)
			
			self.arieal_speed = max(1500, car_vel.len()) + self.arieal_acceleration
			
			time_to_reach_ball = 0
			refine = 0
			
			car_to_ball = 1
			ball_offset = 1
			
			# Calculate impluse vector
			
			while refine < 20.0:
				ball = prediction.slices[clamp(math.ceil(time_to_reach_ball * 60), 0, len(prediction.slices) - 1)]
				
				# Calculate an offset vector to use when steering the car
				vel = Make_Vect(ball.physics.velocity)
				ball_to_target = Make_Vect(ball.physics.location) - aim_pos
				ball_offset = (ball_to_target.normal(max(500, vel.len() * 1.5)) - vel).normal(-70)
				
				car_to_ball = Make_Vect(ball.physics.location) - car_pos + ball_offset
				
				target_vel_xy2 = Vector2(car_to_ball.x, car_to_ball.y)
				
				time_to_reach_ball = car_to_ball.len() / self.arieal_speed
				
				target_vel_xy = target_vel_xy2 * (self.arieal_speed / target_vel_xy2.len())
				
				z_vel = (car_to_ball.z - 0.5 * packet.game_info.world_gravity_z * time_to_reach_ball * time_to_reach_ball) / time_to_reach_ball
				
				# velocity we want to be going
				target_vel = Vec3(target_vel_xy.x, target_vel_xy.y, z_vel)
				
				refine = refine + 1
			
			self.renderer.begin_rendering()
			self.renderer.draw_line_3d(ball_pos.UI_Vec3(), (ball_pos + ball_offset).UI_Vec3(), self.renderer.red())
			self.renderer.draw_line_3d(aim_pos.UI_Vec3(), (aim_pos + Vec3(0, 0, 100)).UI_Vec3(), self.renderer.yellow())
			self.renderer.end_rendering()
			
			impulse = target_vel - car_vel
			
			rot = impulse.to_rotation()
			
			car_rot = car.physics.rotation
			
			car_rot_vel = Make_Vect(car.physics.angular_velocity) #.align_to(car_rot)
			
			car_to_ball_2D = Vector2(ball_pos.x, ball_pos.y) - Vector2(car_pos.x, car_pos.y)
			
			self.controller_state.yaw = correct(rot.x, car_rot.yaw, 0.5)
			self.controller_state.pitch = correct(rot.y + 0.05, car_rot.pitch, 1.0)
			
			self.controller_state.steer = correct(rot.x, car_rot.yaw, 1.0)
			
			a = angle_between(impulse, Rotation_Vector(car_rot))
			pitch_a = constrain_pi(car_rot.pitch - rot.y)
			self.controller_state.boost = impulse.len() > a * 200.0 and abs(pitch_a) < math.cos(rot.y) * car_to_ball_2D.len() / car_to_ball.z * 2.0 and a < math.pi * 0.5
			
			self.controller_state.roll = correct(0.0, car_rot.roll)
			
			self.controller_state.throttle = 1.0
			
			self.controller_state.jump = self.jump_timer > 0.3 and not car.double_jumped and (impulse.z > impulse.len() * 0.5 or impulse.z > 250.0)
			
			# self.renderer.begin_rendering()
			
			# self.renderer.draw_line_3d(car.physics.location, (car_pos + impulse).toVector3(), self.renderer.red())
			# self.renderer.draw_line_3d(car.physics.location, (car_pos + car_to_ball).toVector3(), self.renderer.white())
			
			# self.renderer.end_rendering()
			
			# Need to work on exit condition. However, for now...
			if not car.jumped and car.has_wheel_contact:
				self.is_arieal = False
				self.line_up_time = 0
				self.jump_timer = 0
				if packet.num_tiles > 0:
					if car.team == 0:
						self.target_location = Vector2(0, -4000)
					else:
						self.target_location = Vector2(0, 4000)
				else:
					self.target_location = Vector2(own_goal.location.x, own_goal.location.y) + Vector2(own_goal.direction.x, own_goal.direction.y) * 500
				self.controller_state.jump = False
				exit_condition = False
			
			if self.controller_state.jump:
				self.controller_state.yaw = 0.0
				self.controller_state.pitch = 0.0
				self.controller_state.roll = 0.0
			
			# Flip into the ball
			if not car.double_jumped and car_to_ball.len() < 600.0 and abs(car_to_ball.z - 70) < 150.0 and car_pos.z > 50.0:
				self.controller_state.jump = True
				yaw = self.controller_state.yaw
				self.controller_state.yaw = math.sin(yaw)
				self.controller_state.pitch = -math.cos(yaw)
				self.controller_state.roll = 0.0
				self.flip_timer = 0.9
		
	else:
		
		prediction = self.get_ball_prediction_struct()
		
		car = packet.game_cars[self.index]
		
		car_pos = Make_Vect(car.physics.location)
		car_vel = Make_Vect(car.physics.velocity)
		
		self.arieal_speed = max(800, car_vel.len() + self.arieal_acceleration)
		
		time_to_reach_ball = 0
		refine = 0
		
		car_to_ball = Make_Vect(prediction.slices[0].physics.location) - car_pos
		
		# Calculate impluse vector
		
		while refine < 20.0:
			
			ball_location = Make_Vect(prediction.slices[clamp(math.ceil(time_to_reach_ball * 60), 0, len(prediction.slices) - 1)].physics.location)
			
			car_to_ball = ball_location - car_pos
			
			time_to_reach_ball = car_to_ball.len() / self.arieal_speed
			
			refine = refine + 1
		
		ball_z = ball_location.z - 100.0
		ball_location = Vector2(ball_location.x, ball_location.y)
		car_location = Vector2(car_pos.x, car_pos.y)
		car_direction = get_car_facing_vector(car)
		car_to_target = ball_location - car_location
		steer_correction_radians = car_direction.correction_to(car_to_target) * 3.0
		
		car_to_ball_2D = ball_location - car_location
		car_to_ball_plus_vel = ball_location - car_location - Vector2(car_vel.x, car_vel.y) * 0.75
		car_to_ball_plus_vel_2 = ball_location - car_location - Vector2(car_vel.x, car_vel.y) * 1.25
		
		self.controller_state.throttle = min(1, max(-1, (car_to_ball_2D.len() - max(ball_z * 3.0, 300)) * 0.005))
		
		if self.controller_state.throttle < 0.0:
			steer_correction_radians = -steer_correction_radians
		
		self.controller_state.handbrake = abs(steer_correction_radians) > math.pi * 0.8
		self.controller_state.steer = -max(-1.0, min(1.0, steer_correction_radians))
		self.controller_state.boost = car_to_ball_plus_vel.len() > max(ball_z * 3.5, 1000) and abs(steer_correction_radians) < math.pi * 0.3
		
		if abs(steer_correction_radians) < 0.3 and self.controller_state.throttle > 0.0:
			self.line_up_time += self.delta
		else:
			self.line_up_time = 0
		
		take_off_angle = math.atan2(car_to_ball.z, car_to_ball_plus_vel_2.len())
		
		if self.line_up_time > 0.2 and abs(take_off_angle - math.pi * 0.15) < math.pi * 0.1 and self.controller_state.throttle >= 0.0 and time_to_reach_ball * 30 < car.boost:
			self.controller_state.jump = True
			self.is_arieal = True
		
		# if abs(steer_correction_radians) > math.pi * 0.8 and car_to_ball.len() > 1000.0:
			# Enter_Half_Flip(self, packet)
		
		# self.renderer.begin_rendering()
		
		# self.renderer.draw_line_3d(car.physics.location, (car_pos + car_to_ball).toVector3(), self.renderer.white())
		# self.renderer.draw_string_3d(car.physics.location, 2, 2, str(take_off_angle / math.pi), self.renderer.white())
		
		# self.renderer.end_rendering()
	
	if not packet.game_info.is_round_active:
		self.reset()
	
	return exit_condition

# Used in dropshot to get where to put the ball
def Get_AOI_Offset(self, packet, position):
	position = Vec3(position.x, position.y, 0.0)
	
	sum_position = position.copy()
	
	counter = 0
	total = 0
	
	for goal in self.get_field_info().goals:
		if (Make_Vect(goal.location) - position).len() < 1200.0 and goal.team_num != packet.game_cars[self.index].team:
			sum_position = sum_position + Make_Vect(goal.location) * (packet.dropshot_tiles[counter].tile_state / 3)
			total += 1
		counter += 1
	if total > 1:
		return sum_position * (1 / total)
	else:
		return sum_position


def Strategy_Dropshot(self, packet):
	
	my_car = packet.game_cars[self.index]
	
	ball = packet.game_ball.physics
	
	direction = Vec3(0, 1, 0)
	if my_car.team != 0:
		direction = Vec3(0, -1, 0)
	
	if my_car.is_demolished:
		self.controller_state.boost = False
		self.controller_state.throttle = 0.0
		return
	
	ball_pos = Make_Vect(ball.location)
	
	if packet.game_ball.latest_touch.team == my_car.team:
		self.last_touch.player_name = packet.game_ball.latest_touch.player_name
		self.last_touch.time_seconds = packet.game_ball.latest_touch.time_seconds
	
	# Create a list of cars sorted by time to get to ball
	my_team_cars = []
	my_team_cars_time = []
	my_team_cars_index = []
	i = 0
	while i < packet.num_cars:
		car = packet.game_cars[i]
		if car.team == my_car.team and not car.is_demolished:
			time = Approximate_Time_To_Ball(self.get_ball_prediction_struct(), i, packet, 5, self.arieal_acceleration, not packet.game_info.is_round_active)
			i2 = 0
			while i2 < len(my_team_cars):
				if my_team_cars_time[i2] > time:
					break
				i2 += 1
			my_team_cars.insert(i2, car)
			my_team_cars_time.insert(i2, time)
			my_team_cars_index.insert(i2, i)
		i += 1
	
	# Car that should attack is the first car in the list that has not hit the ball and is facing ball
	attack_car = -1
	for car_index in my_team_cars_index:
		car = packet.game_cars[car_index]
		car_direction = get_car_facing_vector(car)
		car_to_ball = ball_pos - Make_Vect(car.physics.location)
		if not (self.last_touch.player_name == car.name and packet.game_info.seconds_elapsed - self.last_touch.time_seconds < 3.0) and (dot_2D(car_direction, car_to_ball) > 0.0 or car.boost > 80.0) and dot_2D(car_to_ball + direction * 1000, direction):
			attack_car = car_index
			break
	
	car_pos = Make_Vect(my_car.physics.location)
	
	car_pos_2D = Vector2(car_pos.x, car_pos.y)
	
	if attack_car == -1:
		Drive_To(self, packet, ball_pos)
	elif self.index == attack_car:
		t = Approximate_Time_To_Ball(self.get_ball_prediction_struct(), self.index, packet, 5, self.arieal_acceleration, True)
		ball_pos = Make_Vect(Get_Ball_At_T(self.get_ball_prediction_struct(), t).location)
		if t > 3.0:
			if car.boost < 80.0:
				Flip_Drive_To(self, packet, ball_pos)
			else:
				Drive_To(self, packet, ball_pos)
		else:
			# Calculate the target for the bot
			car_to_ball = ball_pos - car_pos
			aim_pos = Get_AOI_Offset(self, packet, car_pos + car_to_ball.normal(car_to_ball.len() + max(ball_pos.z * 1.5, 500)))
			Hit_Ball_To(self, packet, aim_pos)
			self.flip_wait_timer = 0.0
		
	else:
		other_car_i = -1
		for car_index in my_team_cars_index:
			if car_index != attack_car and car_index != self.index:
				other_car_i = car_index
				break
		
		target_location = Vector2(0, 0)
		if other_car_i == -1:
			if car.team == 0:
				target_location = Vector2(0, max(ball.location.y - 3000, -4000))
			else:
				target_location = Vector2(0, min(ball.location.y + 3000, 4000))
		else:
			other_car = packet.game_cars[other_car_i]
			
			l1 = 0
			l2 = 0
			
			other_car_pos = Vector2(other_car.physics.location.x, other_car.physics.location.y)
			
			s = sign(ball.location.x)
			
			if car.team == 0:
				l1 = Vector2(ball.location.x - s * 2500, max(ball.location.y - 2000, -2000))
			else:
				l1 = Vector2(ball.location.x - s * 2500, min(ball.location.y + 2000, 2000))
			
			if car.team == 0:
				l2 = Vector2(constrain(ball.location.x, -3000, 3000), max(ball.location.y - 3000, -4000))
			else:
				l2 = Vector2(constrain(ball.location.x, -3000, 3000), min(ball.location.y + 3000, 4000))
			
			len1 = (l1 - other_car_pos).len()
			len2 = (l2 - other_car_pos).len()
			
			len1_2 = (l1 - car_pos_2D).len()
			len2_2 = (l2 - car_pos_2D).len()
			
			if len1 + len2_2 < len2 + len1_2:
				target_location = l2
			else:
				target_location = l1
			
		# self.renderer.begin_rendering()
		
		# self.renderer.draw_line_3d(my_car.physics.location, target_location.toVector3(), self.renderer.red())
		
		# self.renderer.end_rendering()
		
		if car.boost < 80.0:
			Flip_Drive_To(self, packet, target_location)
		else:
			Drive_To(self, packet, target_location)

class Last_Touch:
	
	def __init__(self):
		self.player_name = ""
		self.time_seconds = -100.0
	

class PenguinBot(BaseAgent):
	
	def reset(self):
		self.flip_timer = 0.0
		self.flip_wait_timer = 0.0
		self.is_arieal = False
		self.line_up_time = 0
		self.target_location = Vector2(1, 1)
		self.jump_timer = 0
		self.match_start = True
		self.has_jump = False
		self.started = False
		self.arieal_speed = 2000.0
		self.arieal_acceleration = 350.0
		self.resetCounter = 0.0
		self.offset = Vec3(0, 0, 0)
		self.delta = 0.0
		self.manuever_lock = -1.0
		self.last_touch = Last_Touch()
	
	def get_total_score(self, packet: GameTickPacket):
		score = 0
		for team in packet.teams:
			score = score + team.score
		return score
	
	def initialize_agent(self):
		#This runs once before the bot starts up
		self.controller_state = SimpleControllerState()
		self.prev_time = 0.0
		self.has_endgame_quickchat = True
		self.reset()
	
	def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
		
		self.controller_state = SimpleControllerState()
		
		self.delta = packet.game_info.seconds_elapsed - self.prev_time
		self.prev_time = packet.game_info.seconds_elapsed
		
		# Celebration and end of game quick chats
		if packet.game_info.is_match_ended:
			my_car = packet.game_cars[self.index]
			
			self.controller_state.boost = my_car.physics.location.z + my_car.physics.velocity.z < 100.0
			self.controller_state.pitch = clamp(-my_car.physics.rotation.pitch + math.pi * 0.4, -1, 1)
			self.controller_state.handbrake = False
			self.controller_state.jump = my_car.has_wheel_contact
			self.controller_state.yaw = 0.0
			
			if my_car.physics.rotation.pitch > math.pi * 0.35:
				self.controller_state.roll = 1.0
			else:
				self.controller_state.roll = 0.0
			
			if self.has_endgame_quickchat:
				self.has_endgame_quickchat = False
				self.send_quick_chat(QuickChats.CHAT_EVERYONE, QuickChats.PostGame_Gg)
				my_score = 0
				other_score = 0
				for team in packet.teams:
					if packet.game_cars[self.index].team == team.team_index:
						my_score = my_score + team.score
					else:
						other_score = other_score + team.score
				
				compliments = [QuickChats.Custom_Compliments_proud, QuickChats.Custom_Compliments_SkillLevel, QuickChats.Custom_Compliments_GC]
				insults = [QuickChats.Custom_Toxic_CatchVirus, QuickChats.Custom_Toxic_404NoSkill, QuickChats.Custom_Toxic_WasteCPU, QuickChats.Custom_Toxic_DeAlloc]
				
				# 
				# Quick chats
				# 
				# For the most part, Penguin bot plays it classy. Penguin bot will compliment the other bot after any win or loss.
				# However, if Penguin bot looses badly, Penguin Bot will become toxic, lashing out with insults worthy of any evil
				# bot that dares to brazil Penguin.
				# 
				
				if abs(my_score - other_score) < 1.5:
					self.send_quick_chat(QuickChats.CHAT_EVERYONE, QuickChats.Reactions_CloseOne)
					if my_score < other_score:
						i = randint(0, len(compliments) - 1)
						self.send_quick_chat(QuickChats.CHAT_EVERYONE, compliments[i])
					
				elif my_score > other_score:
					self.send_quick_chat(QuickChats.CHAT_EVERYONE, QuickChats.PostGame_WhatAGame)
				elif my_score < other_score - 3:
					i = randint(0, len(insults) - 1)
					self.send_quick_chat(QuickChats.CHAT_EVERYONE, insults[i])
				else:
					i = randint(0, len(compliments) - 1)
					self.send_quick_chat(QuickChats.CHAT_EVERYONE, compliments[i])
				
			
		# Match Logic
		else:
			# Strategy_Dropshot(self, packet)
			if self.manuever_lock <= 0.0:
				# Only dropshot tactics for now. Will be updated for threes later
				Strategy_Dropshot(self, packet)
				# Enter_Half_Flip(self, packet)
			else:
				self.manuever_lock = self.manuever_lock - self.delta
				self.manuever(self, packet, self.manuever_lock)
			# Drive_To(self, packet, packet.game_ball.physics.location)
		
		return self.controller_state

class Vector2:
	def __init__(self, x=0, y=0):
		self.x = float(x)
		self.y = float(y)
	
	def __add__(self, val):
		return Vector2(self.x + val.x, self.y + val.y)

	def __sub__(self, val):
		return Vector2(self.x - val.x, self.y - val.y)
		
	def __mul__(self, val):
		return Vector2(self.x * val, self.y * val)
	
	def correction_to(self, ideal):
		# The in-game axes are left handed, so use -x
		current_in_radians = math.atan2(self.y, -self.x)
		ideal_in_radians = math.atan2(ideal.y, -ideal.x)

		correction = ideal_in_radians - current_in_radians

		# Make sure we go the 'short way'
		if abs(correction) > math.pi:
			if correction < 0:
				correction += 2 * math.pi
			else:
				correction -= 2 * math.pi

		return correction
	
	def len(self):
		return math.sqrt(self.x * self.x + self.y * self.y)
	
	def normal(self, n = 1):
		l = max(0.0001, self.len())
		return Vector2(self.x / l * n, self.y / l * n)
	
	def toVector3(self):
		return UI_Vec3(self.x, self.y, 0)

def get_car_facing_vector(car):
	pitch = float(car.physics.rotation.pitch)
	yaw = float(car.physics.rotation.yaw)
	
	# facing_x = math.cos(pitch) * math.cos(yaw)
	# facing_y = math.cos(pitch) * math.sin(yaw)
	
	facing_x = math.cos(yaw)
	facing_y = math.sin(yaw)
	
	return Vector2(facing_x, facing_y)

