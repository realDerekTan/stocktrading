# import useful Python libraries
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib import style
import time
import functools

totalStart = time.time()

style.use("ggplot")

# converts date number to Year-month-day format
def bytedate2num(fmt):
	def converter(b):
		return mdates.strpdate2num(fmt)(b.decode('ascii'))
	return converter

date_converter = bytedate2num("%Y%m%d%H%M%S")

# load text and retrieve date, bid, and ask values
date, bid, ask = np.loadtxt('C:\\Algotrading\\GBPUSD1d.txt', unpack=True,
								delimiter=',',
								converters={0: date_converter})

# calculates percent change between two price points in time
def percentChange(startPoint, currentPoint):
	try:
		x = ((float(currentPoint) - startPoint) / abs(startPoint)) * 100.00
		if x == 0.0:
			return 0.0000000001
		else:
			return x
	except:
		return 0.0000000001

def patternStorage():
	patstartTime = time.time()
	x = len(avgLine) - 60
	y = 31
	while y < x:
		pattern = []
		p1 = percentChange(avgLine[y - 30], avgLine[y - 29])
		p2 = percentChange(avgLine[y - 30], avgLine[y - 28])
		p3 = percentChange(avgLine[y - 30], avgLine[y - 27])
		p4 = percentChange(avgLine[y - 30], avgLine[y - 26])
		p5 = percentChange(avgLine[y - 30], avgLine[y - 25])
		p6 = percentChange(avgLine[y - 30], avgLine[y - 24])
		p7 = percentChange(avgLine[y - 30], avgLine[y - 23])
		p8 = percentChange(avgLine[y - 30], avgLine[y - 22])
		p9 = percentChange(avgLine[y - 30], avgLine[y - 21])
		p10 = percentChange(avgLine[y - 30], avgLine[y - 20])
		p11 = percentChange(avgLine[y - 30], avgLine[y - 19])
		p12 = percentChange(avgLine[y - 30], avgLine[y - 18])
		p13 = percentChange(avgLine[y - 30], avgLine[y - 17])
		p14 = percentChange(avgLine[y - 30], avgLine[y - 16])
		p15 = percentChange(avgLine[y - 30], avgLine[y - 15])
		p16 = percentChange(avgLine[y - 30], avgLine[y - 14])
		p17 = percentChange(avgLine[y - 30], avgLine[y - 13])
		p18 = percentChange(avgLine[y - 30], avgLine[y - 12])
		p19 = percentChange(avgLine[y - 30], avgLine[y - 11])
		p20 = percentChange(avgLine[y - 30], avgLine[y - 10])
		p21 = percentChange(avgLine[y - 30], avgLine[y - 9])
		p22 = percentChange(avgLine[y - 30], avgLine[y - 8])
		p23 = percentChange(avgLine[y - 30], avgLine[y - 7])
		p24 = percentChange(avgLine[y - 30], avgLine[y - 6])
		p25 = percentChange(avgLine[y - 30], avgLine[y - 5])
		p26 = percentChange(avgLine[y - 30], avgLine[y - 4])
		p27 = percentChange(avgLine[y - 30], avgLine[y - 3])
		p28 = percentChange(avgLine[y - 30], avgLine[y - 2])
		p29 = percentChange(avgLine[y - 30], avgLine[y - 1])
		p30 = percentChange(avgLine[y - 30], avgLine[y])

		outcomeRange = avgLine[y+20:y+30]
		currentPoint = avgLine[y]

		try:
			avgOutcome = functools.reduce(lambda x, y:x+y, outcomeRange) / len(outcomeRange)
		except:
			avgOutcome = 0

		futureOutcome = percentChange(currentPoint, avgOutcome)
		pattern.append(p1)
		pattern.append(p2)
		pattern.append(p3)
		pattern.append(p4)
		pattern.append(p5)
		pattern.append(p6)
		pattern.append(p7)
		pattern.append(p8)
		pattern.append(p9)
		pattern.append(p10)
		pattern.append(p11)
		pattern.append(p12)
		pattern.append(p13)
		pattern.append(p14)
		pattern.append(p15)
		pattern.append(p16)
		pattern.append(p17)
		pattern.append(p18)
		pattern.append(p19)
		pattern.append(p20)
		pattern.append(p21)
		pattern.append(p22)
		pattern.append(p23)
		pattern.append(p24)
		pattern.append(p25)
		pattern.append(p26)
		pattern.append(p27)
		pattern.append(p28)
		pattern.append(p29)
		pattern.append(p30)

		patternAr.append(pattern)
		performanceAr.append(futureOutcome)

		print('Where we are historically:',currentPoint)
		print('Soft outcome on the horizon:', avgOutcome)
		print('This pattern brings a future change of:', futureOutcome)
		print(' _____')
		print(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10)
		y += 1

	patEndTime = time.time()
	print(len(patternAr))
	print(len(performanceAr))
	timeUsed = patEndTime - patstartTime
	print('Pattern storage took:', timeUsed, 'seconds')

def currentPattern():

	# calculate percentage change for every specific time frame
	cp1 = percentChange(avgLine[-31], avgLine[-30])
	cp2 = percentChange(avgLine[-31], avgLine[-29])
	cp3 = percentChange(avgLine[-31], avgLine[-28])
	cp4 = percentChange(avgLine[-31], avgLine[-27])
	cp5 = percentChange(avgLine[-31], avgLine[-26])
	cp6 = percentChange(avgLine[-31], avgLine[-25])
	cp7 = percentChange(avgLine[-31], avgLine[-24])
	cp8 = percentChange(avgLine[-31], avgLine[-23])
	cp9 = percentChange(avgLine[-31], avgLine[-22])
	cp10 = percentChange(avgLine[-31], avgLine[-21])
	cp11 = percentChange(avgLine[-31], avgLine[-20])
	cp12 = percentChange(avgLine[-31], avgLine[-19])
	cp13 = percentChange(avgLine[-31], avgLine[-18])
	cp14 = percentChange(avgLine[-31], avgLine[-17])
	cp15 = percentChange(avgLine[-31], avgLine[-16])
	cp16 = percentChange(avgLine[-31], avgLine[-15])
	cp17 = percentChange(avgLine[-31], avgLine[-14])
	cp18 = percentChange(avgLine[-31], avgLine[-13])
	cp19 = percentChange(avgLine[-31], avgLine[-12])
	cp20 = percentChange(avgLine[-31], avgLine[-11])
	cp21 = percentChange(avgLine[-31], avgLine[-10])
	cp22 = percentChange(avgLine[-31], avgLine[-9])
	cp23 = percentChange(avgLine[-31], avgLine[-8])
	cp24 = percentChange(avgLine[-31], avgLine[-7])
	cp25 = percentChange(avgLine[-31], avgLine[-6])
	cp26 = percentChange(avgLine[-31], avgLine[-5])
	cp27 = percentChange(avgLine[-31], avgLine[-4])
	cp28 = percentChange(avgLine[-31], avgLine[-3])
	cp29 = percentChange(avgLine[-31], avgLine[-2])
	cp30 = percentChange(avgLine[-31], avgLine[-1])

	# append all data to the pattern recognition array
	patForRec.append(cp1)
	patForRec.append(cp2)
	patForRec.append(cp3)
	patForRec.append(cp4)
	patForRec.append(cp5)
	patForRec.append(cp6)
	patForRec.append(cp7)
	patForRec.append(cp8)
	patForRec.append(cp9)
	patForRec.append(cp10)
	patForRec.append(cp11)
	patForRec.append(cp12)
	patForRec.append(cp13)
	patForRec.append(cp14)
	patForRec.append(cp15)
	patForRec.append(cp16)
	patForRec.append(cp17)
	patForRec.append(cp18)
	patForRec.append(cp19)
	patForRec.append(cp20)
	patForRec.append(cp21)
	patForRec.append(cp22)
	patForRec.append(cp23)
	patForRec.append(cp24)
	patForRec.append(cp25)
	patForRec.append(cp26)
	patForRec.append(cp27)
	patForRec.append(cp28)
	patForRec.append(cp29)
	patForRec.append(cp30)

	print(patForRec)

def patternRecognition():

	predictedOutcomesAr = []
	patFound = 0
	plotPatAr = []

	# calculate similarity between patterns, 100% being full match, 0% being no match
	for eachPattern in patternAr:
		sim1 = float(100) - abs(percentChange(eachPattern[0], patForRec[0]))
		sim2 = float(100) - abs(percentChange(eachPattern[1], patForRec[1]))
		sim3 = float(100) - abs(percentChange(eachPattern[2], patForRec[2]))
		sim4 = float(100) - abs(percentChange(eachPattern[3], patForRec[3]))
		sim5 = float(100) - abs(percentChange(eachPattern[4], patForRec[4]))
		sim6 = float(100) - abs(percentChange(eachPattern[5], patForRec[5]))
		sim7 = float(100) - abs(percentChange(eachPattern[6], patForRec[6]))
		sim8 = float(100) - abs(percentChange(eachPattern[7], patForRec[7]))
		sim9 = float(100) - abs(percentChange(eachPattern[8], patForRec[8]))
		sim10 = float(100) - abs(percentChange(eachPattern[9], patForRec[9]))
		sim11 = float(100) - abs(percentChange(eachPattern[10], patForRec[10]))
		sim12 = float(100) - abs(percentChange(eachPattern[11], patForRec[11]))
		sim13 = float(100) - abs(percentChange(eachPattern[12], patForRec[12]))
		sim14 = float(100) - abs(percentChange(eachPattern[13], patForRec[13]))
		sim15 = float(100) - abs(percentChange(eachPattern[14], patForRec[14]))
		sim16 = float(100) - abs(percentChange(eachPattern[15], patForRec[15]))
		sim17 = float(100) - abs(percentChange(eachPattern[16], patForRec[16]))
		sim18 = float(100) - abs(percentChange(eachPattern[17], patForRec[17]))
		sim19 = float(100) - abs(percentChange(eachPattern[18], patForRec[18]))
		sim20 = float(100) - abs(percentChange(eachPattern[19], patForRec[19]))
		sim21 = float(100) - abs(percentChange(eachPattern[20], patForRec[20]))
		sim22 = float(100) - abs(percentChange(eachPattern[21], patForRec[21]))
		sim23 = float(100) - abs(percentChange(eachPattern[22], patForRec[22]))
		sim24 = float(100) - abs(percentChange(eachPattern[23], patForRec[23]))
		sim25 = float(100) - abs(percentChange(eachPattern[24], patForRec[24]))
		sim26 = float(100) - abs(percentChange(eachPattern[25], patForRec[25]))
		sim27 = float(100) - abs(percentChange(eachPattern[26], patForRec[26]))
		sim28 = float(100) - abs(percentChange(eachPattern[27], patForRec[27]))
		sim29 = float(100) - abs(percentChange(eachPattern[28], patForRec[28]))
		sim30 = float(100) - abs(percentChange(eachPattern[29], patForRec[29]))

		howSim = (sim1 + sim2 + sim3 + sim4 + sim5 + sim6 + sim7 + sim8 + sim9 + sim10 + sim11 + sim12 + sim13 +
				  sim14 + sim15 + sim16 + sim17 + sim18 + sim19 + sim20 + sim21 + sim22 + sim23 + sim24 + sim25 +
				  sim26 + sim27 + sim28 + sim29 + sim30) / \
				 float(30)

		if howSim > 70:
			patdex = patternAr.index(eachPattern)

			patFound = 1

			xp = range(1, 31)
			plotPatAr.append(eachPattern)

	predArray = []
	if patFound == 1:
		# fig = plt.figure(figsize=(10, 6))

		for eachPatt in plotPatAr:
			futurePoints = patternAr.index(eachPatt)

			if performanceAr[futurePoints] > patForRec[29]:
				pcolor = '#24bc00'
				predArray.append(1.000)
			else:
				pcolor = '#d40000'
				predArray.append(-1.000)

			# plt.plot(xp, eachPatt)
			predictedOutcomesAr.append(performanceAr[futurePoints])

			# plt.scatter(35, performanceAr[futurePoints], c = pcolor, alpha = .3)

		realOutcomeRange = allData[toWhat+20:toWhat+30]
		realAvgOutcome = functools.reduce(lambda x, y:x+y, realOutcomeRange) / len(realOutcomeRange)
		realMovement = percentChange(allData[toWhat], realAvgOutcome)
		predictedAvgOutcome = functools.reduce(lambda x, y: x + y, predictedOutcomesAr) / len(predictedOutcomesAr)

		print(predArray)
		predictionAverage = functools.reduce(lambda x, y: x + y, predArray) / len(predArray)

		print(predictionAverage)

		if predictionAverage < 0:
			print('Drop predicted')
			print(patForRec[29])
			print(realMovement)
			if realMovement < patForRec[29]:
				accuracyArray.append(100)
			else:
				accuracyArray.append(0)

		if predictionAverage > 0:
			print('Rise predicted')
			print(patForRec[29])
			print(realMovement)
			if realMovement < patForRec[29]:
				accuracyArray.append(100)
			else:
				accuracyArray.append(0)

	# plt.scatter(40, realMovement, c='#54fff7', s=25)
		# plt.scatter(40, predictedAvgOutcome, c='b', s=25)

		# plt.plot(xp, patForRec, '#54fff7', linewidth = 3)
		# plt.grid(True)
		# plt.title('Pattern Recognition')
		# plt.show()

# graph price chart for data
def graphRawFX():

	fig = plt.figure(figsize=(10, 7))

	ax1 = plt.subplot2grid((40, 40), (0, 0), rowspan=40, colspan=40)
	ax1.plot(date, bid)
	ax1.plot(date, ask)
	plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)

	ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))

	for label in ax1.xaxis.get_ticklabels():
		label.set_rotation(45)

	ax1_2 = ax1.twinx()
	ax1_2.fill_between(date, 0, (ask-bid), facecolor='g', alpha=.3)

	plt.subplots_adjust(bottom=.23)
	plt.grid(True)
	plt.show()

dataLength = int(bid.shape[0])
print('Data length is:', dataLength)

toWhat = 55000
allData = ((bid + ask) / 2)

avgLine = ((bid + ask) / 2)

# store saved patterns and performance in an array
patternAr = []
performanceAr = []
patternStorage()

accuracyArray = []
samps = 0

while toWhat < dataLength:
	avgLine = allData[:toWhat]

	patForRec = []

	currentPattern()
	patternRecognition()

	totalEnd = time.time() - totalStart
	print(accuracyArray)
	accuracyAverage = functools.reduce(lambda x, y: x+y, accuracyArray) / len(accuracyArray)
	samps += 1
	toWhat += 1

	print('Entire processing time took:', totalEnd, 'seconds')
	print('Backtested Accuracy is', str(accuracyAverage)+'% after', samps, 'actionable trades')
