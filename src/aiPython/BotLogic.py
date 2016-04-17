import zerorpc
import sys
import numpy as np
import json
import math
import itertools
import time
import matplotlib.pyplot as plt

import tensorflow as tf
from tf_rl.models import MLP
from tf_rl.controller import DiscreteDeepQ

class HelloRPC(object):
	def __init__(self):
		self.samples = []
		self.lastNodeMass = 10
		self.lastState = None
		self.lastTargetAngle = None
		self.numCellsInStateVector = 1
		
		self.numiterations = 0

		self.session = tf.InteractiveSession()
		self.brain = MLP([2*self.numCellsInStateVector,],[200, 200, 4], [tf.tanh, tf.tanh, tf.identity])
		self.optimizer = tf.train.RMSPropOptimizer(learning_rate= 0.001, decay=0.9)
		# DiscreteDeepQ object
		self.current_controller = DiscreteDeepQ(2*self.numCellsInStateVector, 4, self.brain,
    		self.optimizer, self.session,
            discount_rate=0.99, exploration_period=5000, 
            max_experience=10000, store_every_nth=1, 
            train_every_nth=100)


		self.session.run(tf.initialize_all_variables())
		self.session.run(self.current_controller.target_network_update)

		self.width = 600
		self.height = 600
		plt.ion()
		self.fig = plt.figure()
		self.ax = self.fig.add_subplot(111)
		self.cell1, = self.ax.plot((300),(300),'o', color='black')

		self.ax.set_xlim(0,600)
		self.ax.set_ylim(0,600)

	def getNewMousePosition(self, currentInfo):
		tick = time.clock()
		currentInfo = json.loads(currentInfo)
		self.numiterations += 1
		nextState = self.createStateFromInfo(currentInfo)

		self.cell1.set_xdata((currentInfo['cell'][0]['position']['x']))
		self.cell1.set_ydata((currentInfo['cell'][0]['position']['y']))
		self.fig.canvas.draw()
		plt.show()
		# print self.numiterations
		if (self.lastState is not None):
			

			self.current_controller.store(
				self.lastState,
				self.lastTargetAngle,
				self.getMassIncrease(currentInfo['cell']),
				nextState
			)

			newAction = self.current_controller.action(nextState)

			self.current_controller.training_step()
		
			targetX, targetY = self.getMousePosFromMove(currentInfo['cell'], newAction)
			self.lastState = nextState
			self.lastTargetAngle = newAction
			self.lastNodeMass = self.getCurrentMass(currentInfo['cell'])

			return {'x': targetX, 'y': targetY, 'message':'time to loop: ' + str(time.clock()-tick)}
			# return {'x': 0, 'y': 0, 'message':'placeholder'}

		else:
			self.lastTargetAction = 0
			self.lastState = self.createStateFromInfo(currentInfo)
			self.lastNodeMass = self.getCurrentMass(currentInfo['cell'])
			lastState = currentInfo
			targetX, targetY = self.getMousePosFromMove(currentInfo['cell'], 0)
			return {'x': targetX, 'y': targetY, 'message':'placeholder'}


		# closest = self.findClosest(currentInfo['cell'], currentInfo['nodes'])
		# if closest:
		# 	return {'x': closest['position']['x'],'y': closest['position']['y'], 'message':closest}
		# else:
		# 	return {'x':0,'y':0, 'message': 'no food in view'}

	def findClosest(self, cell, otherNodes):
		closestDist = 100000
		closestNode = None
		a = np.array((cell['position']['x'], cell['position']['y']))
		# return otherNodes[0]
		for node in otherNodes:
			# return node
			b = np.array((node['position']['x'], node['position']['y']))

			dist = np.linalg.norm(a-b)
			if dist<closestDist and node['cellType'] == 1:
				closestDist = dist
				closestNode = node

		return closestNode

	def createStateFromInfo(self, currentInfo):
		#Here, we're going to transform the current info into a state array
		#we'll include the following information for all of the cells
		# angle, distance, size
		# We're including the 100 closest items to the bot. 
		cellXPos, cellYPos = self.getCenterPos(currentInfo['cell'])
		cellPos = np.array((cellXPos, cellYPos))
		cellSize = self.getAverageNodeMass(currentInfo['cell'])
		otherCellInfo = []

		for otherCell in currentInfo['nodes']:
			if (otherCell['cellType'] == 1):
				distX = otherCell['position']['x'] - cellXPos
				distY = otherCell['position']['y'] - cellYPos
				# angle = int(math.degrees(math.atan2((otherCell['position']['y'] - cellYPos),(otherCell['position']['x'] - cellXPos))))
				# angle = (360 + angle) % 360
				# otherCellPos = np.array((otherCell['position']['x'],otherCell['position']['y']))
				# distance = np.linalg.norm(otherCellPos-cellPos)
				sizeRatio = otherCell['mass']/cellSize
				otherCellType = otherCell['cellType']
				otherCellInfo.append((float(distX)/self.width, float(distY)/self.height))
				# otherCellInfo.append((angle, distance))

				# otherCellInfo.append((angle))

		topCells = sorted(otherCellInfo, key=lambda cell: cell[1])
		# if (len(otherCellInfo) == 0):
		# 	otherCellInfo.append(np.random.randint(0,high=359,size=1))
		# topCells = sorted(otherCellInfo)
		stateCells = topCells[0:self.numCellsInStateVector]
		if (len(stateCells) < self.numCellsInStateVector):
			fakeCells = [(1, 1)] * (self.numCellsInStateVector-len(stateCells))
			stateCells = stateCells + fakeCells

		stateCells = np.array(list(itertools.chain(*stateCells)))

		# stateCells = np.array(stateCells)
		# stateCells = stateCells.reshape(stateCells.shape[0],1)
		return stateCells

	def getAverageNodeMass(self, cell):
		massList = []
		for part in cell:
			massList.append(part['mass'])
		return sum(massList)/float(len(massList))

	def getCurrentMass(self, cell):
		massList = []
		for part in cell:
			massList.append(part['mass'])
		return sum(massList)

	def getMassIncrease(self, cell):
		newMass = self.getCurrentMass(cell)
		return (newMass - self.lastNodeMass)/self.lastNodeMass

	def getMousePosFromMove(self, cell, move):
		radius = 10

		xPos, yPos = self.getCenterPos(cell)
		newX = xPos
		newY = yPos


		if (move == 0):
			newX += radius
		elif (move == 1):
			newY += radius
		elif (move == 2):
			newX -= radius
		else:
			newY -= radius

		return newX, newY

	def getCenterPos (self, cell):
		xs = []
		ys = []

		for part in cell:
			xs.append(part['position']['x']) 
			ys.append(part['position']['y'])

		xPos = sum(xs)/(float(len(xs)))
		yPos = sum(ys)/(float(len(ys)))

		return xPos, yPos


s = zerorpc.Server(HelloRPC())
print "python port", str(sys.argv[1])

s.bind("tcp://0.0.0.0:"+str(sys.argv[1]))
s.run()