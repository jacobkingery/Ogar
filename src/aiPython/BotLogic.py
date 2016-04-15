import zerorpc
import sys
import numpy as np
import json
import lspi
import math
import itertools

from pybrain.tools.shortcuts import buildNetwork

class HelloRPC(object):
	def __init__(self):
		self.samples = []
		self.previousMass = 10;
		self.numCellsInStateVector = 1
		self.numiterations = 0
		self.net = buildNetwork(2, 3, 1)



	def getNewMousePosition(self, currentInfo):
		currentInfo = json.loads(currentInfo)
		currentState = self.createStateFromInfo(currentInfo)
		self.numiterations += 1

		angle = 0
		bestReward = -100000000;
		for i in range(360):
			reward = self.net.activate(i,currentState[:1])
			if reward > bestReward:
				angle = i
				bestReward = reward

		angle = self.net.activate(currentState[:1])
		targetX, targetY = self.getMousePosFromAngle(currentInfo['cell'], angle)
		# print currentState
		angleString = 'angle: '+ str(angle)
		return {'x': targetX, 'y': targetY, 'message':angleString}


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

		# print stateCells
		stateCells = np.array(list(itertools.chain(*stateCells)))
		return stateCells
		# stateCells = np.array(stateCells)
		# stateCells = stateCells.reshape(stateCells.shape[0],1)
		# return stateCells

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