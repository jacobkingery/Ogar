import zerorpc
import sys
import numpy as np
import json
import lspi
import math
import itertools


class HelloRPC(object):
	def __init__(self):
		self.samples = []
		self.lastState = None
		self.lastTargetAngle = None
		self.numCellsInStateVector = 10
		self.lspiLearner = lspi.lspi
		self.policy = lspi.policy.Policy(
			lspi.basis_functions.RadialBasisFunction(
				[np.zeros((1,4*self.numCellsInStateVector))],
				1,
				360,
			),
			discount=0.9, 
			explore=0.4
		)

	def getNewMousePosition(self, currentInfo):
		currentInfo = json.loads(currentInfo)
		if (self.lastState is not None):
			print "getting new sample"
			nextState = self.createStateFromInfo(currentInfo)
			newSample = lspi.sample.Sample(
				self.lastState,
				self.lastTargetAngle,
				self.getAverageNodeSize(currentInfo['cell']),
				nextState
			)

			self.samples.append(newSample)
			print "learning"
			# learn with the new samples!
			self.lspiLearner.learn(
				self.samples, 
				self.policy,
				lspi.solvers.LSTDQSolver(
					precondition_value=0.99
				),
				max_iterations=2

			)
			print "done"

			newAction = self.policy.best_action(nextState)

			targetX, targetY = self.getMousePosFromAngle(currentInfo['cell'], newAction)
			self.lastState = nextState
			self.lastTargetAngle = newAction
			return {'x': targetX, 'y': targetY, 'message':'placeholder'}
			return {'x': 0, 'y': 0, 'message':'placeholder'}

		else:
			self.lastTargetAngle = 0
			self.lastState = self.createStateFromInfo(currentInfo)
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
		cellSize = self.getAverageNodeSize(currentInfo['cell'])
		otherCellInfo = []

		for otherCell in currentInfo['nodes']:
			angle = int(math.degrees(math.atan((otherCell['position']['x'] - cellXPos)/(otherCell['position']['y'] - cellYPos))))
			otherCellPos = np.array((otherCell['position']['x'],otherCell['position']['y']))
			distance = np.linalg.norm(cellPos-otherCellPos)
			sizeRatio = otherCell['mass']/cellSize
			otherCellType = otherCell['cellType']
			otherCellInfo.append((angle, distance, sizeRatio, otherCellType))

		topCells = sorted(otherCellInfo, key=lambda cell: cell[1])

		stateCells = topCells[0:self.numCellsInStateVector]
		if (len(stateCells) < self.numCellsInStateVector):
			fakeCells = [(0, 1000000, 0, 0)] * (self.numCellsInStateVector-len(stateCells))
			stateCells = stateCells + fakeCells

		stateCells = np.array(list(itertools.chain(*stateCells)))
		stateCells = stateCells.reshape(1,stateCells.shape[0])
		print stateCells.shape
		return stateCells

	def getAverageNodeSize(self, cell):
		sizeList = []
		for part in cell:
			sizeList.append(part['mass'])
		return sum(sizeList)/float(len(sizeList))

	def getMousePosFromAngle(self, cell, angle):
		radius = 10
		
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