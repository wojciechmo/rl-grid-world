import numpy as np
import cv2
import sys

class Environment(object):
	
	def __init__(self, gridH, gridW, end_positions, end_rewards, blocked_positions, start_position, default_reward, scale=100):
		
		self.action_space = 4
		self.state_space = gridH * gridW	
		self.gridH = gridH
		self.gridW = gridW
		self.scale = scale 

		self.end_positions = end_positions
		self.end_rewards = end_rewards
		self.blocked_positions = blocked_positions
		
		self.start_position = start_position
		if self.start_position == None:
			self.position = self.init_start_state()
		else:
			self.position = self.start_position
						
		self.state2idx = {}
		self.idx2state = {}
		self.idx2reward = {}
		for i in range(self.gridH):
			for j in range(self.gridW):
				idx = i*self.gridW + j
				self.state2idx[(i, j)] = idx
				self.idx2state[idx]=(i, j)
				self.idx2reward[idx] = default_reward
				
		for position, reward in zip(self.end_positions, self.end_rewards):
			self.idx2reward[self.state2idx[position]] = reward

		self.frame = np.zeros((self.gridH * self.scale, self.gridW * self.scale, 3), np.uint8)	
		
		for position in self.blocked_positions:
			
			y, x = position
			
			cv2.rectangle(self.frame, (x*self.scale, y*self.scale), ((x+1)*self.scale, (y+1)*self.scale), (100, 100, 100), -1)
		
		for position, reward in zip(self.end_positions, self.end_rewards):

			text = str(int(reward))
			if reward > 0.0: text = '+' + text
			
			if reward > 0.0: color = (0, 255, 0)
			else: color = (0, 0, 255)
			font = cv2.FONT_HERSHEY_SIMPLEX
			y, x = position		
			(w, h), _ = cv2.getTextSize(text, font, 1, 2)
				
			cv2.putText(self.frame, text, (int((x+0.5)*self.scale)-w/2, int((y+0.5)*self.scale+h/2)), font, 1, color, 2, cv2.LINE_AA)

	def init_start_state(self):
		
		while True:
			
			preposition = (np.random.choice(self.gridH), np.random.choice(self.gridW))
			
			if preposition not in self.end_positions and preposition not in self.blocked_positions:
				
				return preposition

	def get_state(self):
		
		return self.state2idx[self.position]
	
	def get_possible_actions(self):
		
		'''
		pos = self.position
		possible_actions = []
		
		if pos[0]+1<self.nH: and (pos[0]+1,pos[1]) not in self.blocked_positions:
			possible_actions.append(0)
				
		if pos[0]-1>=0 and (pos[0]-1,pos[1]) not in self.blocked_positions:
			possible_actions.append(1)
		
		if pos[1]+1<self.nW and (pos[0],pos[1]+1) not in self.blocked_positions:
			possible_actions.append(2)
				
		if pos[1]-1>=0 and (pos[0],pos[1]-1) not in self.blocked_positions:
			possible_actions.append(3)
		
		return possible_actions		
		'''
		
		return range(self.action_space) 
	
	def step(self, action):
		
		if action >= self.action_space:
			return

		if action == 0:
			proposed = (self.position[0] +1, self.position[1])
			
		elif action == 1:
			proposed = (self.position[0] -1, self.position[1])
			
		elif action == 2:
			proposed = (self.position[0], self.position[1] +1)
			
		elif action == 3:
			proposed = (self.position[0], self.position[1] -1)	
		
		y_within = proposed[0] >= 0 and proposed[0] < self.gridH
		x_within = proposed[1] >= 0 and proposed[1] < self.gridW
		free = proposed not in self.blocked_positions		
		
		if x_within and y_within and free:
			
			self.position = proposed
			
		next_state = self.state2idx[self.position] 
		reward = self.idx2reward[next_state]
		
		if self.position in self.end_positions:
			done = True
		else:
			done = False
			
		return next_state, reward, done
		
	def reset_state(self):
		
		if self.start_position == None:
			self.position = self.init_start_state()
		else:
			self.position = self.start_position
	
	def render(self, qvalues_matrix):
		
		frame = self.frame.copy()
		
		# for each state cell
		
		for idx, qvalues in enumerate(qvalues_matrix):
			
			position = self.idx2state[idx]
		
			if position in self.end_positions or position in self.blocked_positions:
				continue
			
			qvalues = np.tanh(qvalues*0.1) # for vizualization only
        	
        	# for each action in state cell	
        		
			for action, qvalue in enumerate(qvalues):

				# draw (state, action) qvalue traingle
				
				if action == 0:
					dx2, dy2, dx3, dy3 = 0.0, 1.0, 1.0, 1.0				
				if action == 1:
					dx2, dy2, dx3, dy3 = 0.0, 0.0, 1.0, 0.0				
				if action == 2:
					dx2, dy2, dx3, dy3 = 1.0, 0.0, 1.0, 1.0				
				if action == 3:
					dx2, dy2, dx3, dy3 = 0.0, 0.0, 0.0, 1.0	
					
				x1 = int(self.scale*(position[1] + 0.5))			
				y1 = int(self.scale*(position[0] + 0.5))				
				
				x2 = int(self.scale*(position[1] + dx2))
				y2 = int(self.scale*(position[0] + dy2))
				
				x3 = int(self.scale*(position[1] + dx3))
				y3 = int(self.scale*(position[0] + dy3))		
				
				pts = np.array([[x1, y1], [x2, y2], [x3, y3]], np.int32)
				pts = pts.reshape((-1, 1, 2))
				
				if qvalue > 0: color = (0, int(qvalue*255),0)
				elif qvalue < 0: color = (0,0, -int(qvalue*255))
				else: color = (0, 0, 0)

				cv2.fillPoly(frame, [pts], color)
			
			# draw crossed lines
			
			x1 = int(self.scale*(position[1]))			
			y1 = int(self.scale*(position[0]))
			
			x2 = int(self.scale*(position[1] + 1.0))			
			y2 = int(self.scale*(position[0] + 1.0))

			x3 = int(self.scale*(position[1]+ 1.0 ))			
			y3 = int(self.scale*(position[0]))
			
			x4 = int(self.scale*(position[1]))			
			y4 = int(self.scale*(position[0] + 1.0))
			
			cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
			cv2.line(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
			
			# draw arrow indicating best action
						
			best_action = 0
			best_qvalue = qvalues[0]
			for action, qvalue in enumerate(qvalues):
				if qvalue > best_qvalue:
					best_qvalue = qvalue
					best_action = action
			
			if best_action == 0:
				dx1, dy1, dx2, dy2 = 0.0, -0.25, 0.0, 0.25	
							
			elif best_action == 1:
				dx1, dy1, dx2, dy2 = 0.0, 0.25, 0.0, -0.25
								
			elif best_action == 2:
				dx1, dy1, dx2, dy2 = -0.25, 0.0, 0.25, 0.0
								
			elif best_action == 3:
				dx1, dy1, dx2, dy2 = 0.25, 0.0, -0.25, 0.0				
									
			x1 = int(self.scale*(position[1] + 0.5 + dx1))			
			y1 = int(self.scale*(position[0] + 0.5 + dy1))	
					
			x2 = int(self.scale*(position[1] + 0.5 + dx2))			
			y2 = int(self.scale*(position[0] + 0.5 + dy2))	
							
			cv2.arrowedLine(frame, (x1, y1), (x2, y2), (255, 100, 0), 8, line_type=8, tipLength=0.5)		
			
		# draw horizontal lines
		
		for i in range(self.gridH+1):
			cv2.line(frame, (0, i*self.scale), (self.gridW * self.scale, i*self.scale), (255, 255, 255), 2)
		
		# draw vertical lines
		
		for i in range(self.gridW+1):
			cv2.line(frame, (i*self.scale, 0), (i*self.scale, self.gridH * self.scale), (255, 255, 255), 2)
					
		# draw agent
		
		y, x = self.position
		
		y1 = int((y + 0.3)*self.scale)
		x1 = int((x + 0.3)*self.scale)
		y2 = int((y + 0.7)*self.scale)
		x2 = int((x + 0.7)*self.scale)

		cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), -1)

		cv2.imshow('frame', frame)
		cv2.moveWindow('frame', 0, 0)
		key = cv2.waitKey(1)
		if key == 27: sys.exit()
