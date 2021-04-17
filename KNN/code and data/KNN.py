import csv
filename = 'wisc_bc_data.csv'
filearray = []
with open(filename) as f:
	f.readline()
	reader = csv.reader(f)
	for row in reader:
		filearray.append(row)
meanlist = []
for i in range(2, 32):
	column = []
	for a in filearray:
		column.append(float(a[i]))
	meanlist.append(sum(column[0:500])/500)
stdev = []
for i in range(2, 32):
	column = []
	for a in filearray:
		column.append(float(a[i]))
	stdev.append((sum((b-meanlist[i-2])**2 for b in column[0:500])/500)**0.5)
stdarray = []#standardize training and testing data
for a in filearray:
	row = []
	for i in range(2, 32):
		row.append((float(a[i])-meanlist[i-2])/stdev[i-2])
	stdarray.append(row)
predict = []
for i in range(500, 569):
	distance = []
	for a in stdarray[0:500]:
		distance.append(sum((a[j]-stdarray[i][j])**2 for j in range(30)))
	sortdistance = sorted(distance)
	index = distance.index(sortdistance[0])#k = 1
	predict.append(filearray[index][1])
count = 0
for i in range(500, 569):
	if filearray[i][1] == predict[i-500]:
		count += 1
accuracy = count/69
print('accuracy: {:.2%}'.format(accuracy))