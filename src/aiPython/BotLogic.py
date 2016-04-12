import zerorpc
import sys
import numpy as np
import json
import lspi
import math
import itertools

class simpleBasis(lspi.basis_functions.BasisFunction):

	def __init__(self, state_size, num_actions):
		self.stateSize = state_size
		self.__num_actions = lspi.basis_functions.BasisFunction._validate_num_actions(num_actions)

	def size(self):
		return self.stateSize + 1

	def evaluate(self, state, action):
		phi = np.array([action, state[0], state[1]])
		return phi

	@property
	def num_actions(self):
		return self.__num_actions

	@num_actions.setter
	def num_actions(self, value):
		if value < 1:
			raise ValueError('num_actions must be at least 1.')
		self.__num_actions = value

class HelloRPC(object):
	def __init__(self):
		self.samples = []
		self.lastNodeMass = 10
		self.lastState = None
		self.lastTargetAngle = None
		self.numCellsInStateVector = 1
		self.lspiLearner = lspi.lspi
		self.numiterations = 0
		# self.policy = lspi.policy.Policy(
		# 	lspi.basis_functions.RadialBasisFunction(
		# 		[np.zeros((1,4*self.numCellsInStateVector))],
		# 		1,
		# 		360,
		# 	),
		# 	discount=0.9, 
		# 	explore=0.4
		# )
		# self.policy = lspi.policy.Policy(
		# 	lspi.basis_functions.FakeBasis(
		# 		360,
		# 	),
		# 	discount=0.9, 
		# 	explore=0.4
		# )
		self.policy = lspi.policy.Policy(
			simpleBasis(
				2*self.numCellsInStateVector,
				360,
			),
			discount=0.5, 
			explore=0.2
		)


	def getNewMousePosition(self, currentInfo):
		currentInfo = json.loads(currentInfo)
		self.numiterations += 1
		# print self.numiterations
		if (self.lastState is not None):

			# print "state", self.lastState
			# print "target angle", self.lastTargetAngle
			# print "reward", self.getMassIncrease(currentInfo['cell'])

			nextState = self.createStateFromInfo(currentInfo)
			newSample = lspi.sample.Sample(
				self.lastState,
				self.lastTargetAngle,
				self.getMassIncrease(currentInfo['cell']),
				nextState
			)

			self.samples.append(newSample)
			# learn with the new samples!
			self.policy = self.lspiLearner.learn(
				[newSample], 
				self.policy,
				lspi.solvers.LSTDQSolver(
					precondition_value=0.5
				),
				max_iterations=10

			)

			newAction = self.policy.best_action(nextState)

			targetX, targetY = self.getMousePosFromAngle(currentInfo['cell'], newAction)
			self.lastState = nextState
			self.lastTargetAngle = newAction
			self.lastNodeMass = self.getCurrentMass(currentInfo['cell'])

			return {'x': targetX, 'y': targetY, 'message':'Difference Between Target Angle and Food: ' + str(self.lastState[0] - self.lastTargetAngle) + ", distance between: " + str(self.lastState[1])}
			# return {'x': 0, 'y': 0, 'message':'placeholder'}

		else:
			self.lastTargetAngle = 0
			self.lastState = self.createStateFromInfo(currentInfo)
			self.lastNodeMass = self.getCurrentMass(currentInfo['cell'])
			lastState = currentInfo
			targetX, targetY = self.getMousePosFromAngle(currentInfo['cell'], 0)
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
				angle = int(math.degrees(math.atan2((otherCell['position']['y'] - cellYPos),(otherCell['position']['x'] - cellXPos))))
				angle = (360 + angle) % 360
				otherCellPos = np.array((otherCell['position']['x'],otherCell['position']['y']))
				distance = np.linalg.norm(otherCellPos-cellPos)
				sizeRatio = otherCell['mass']/cellSize
				otherCellType = otherCell['cellType']
				# otherCellInfo.append((angle, distance, sizeRatio, otherCellType))
				otherCellInfo.append((angle, distance))

				# otherCellInfo.append((angle))

		topCells = sorted(otherCellInfo, key=lambda cell: cell[1])
		# if (len(otherCellInfo) == 0):
		# 	otherCellInfo.append(np.random.randint(0,high=359,size=1))
		# topCells = sorted(otherCellInfo)
		stateCells = topCells[0:self.numCellsInStateVector]
		if (len(stateCells) < self.numCellsInStateVector):
			fakeCells = [(0, 1000000, 0, 0)] * (self.numCellsInStateVector-len(stateCells))
			stateCells = stateCells + fakeCells

		stateCells = np.array(list(itertools.chain(*stateCells)))

		stateCells = np.array(stateCells)
		stateCells = stateCells.reshape(stateCells.shape[0],1)
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

	def getMousePosFromAngle(self, cell, angle):
		radius = 100
		
		xPos, yPos = self.getCenterPos(cell)

		newX = xPos + radius*(math.cos(math.radians(angle)))
		newY = yPos + radius*(math.sin(math.radians(angle)))

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