import kNN
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#a = np.loadtxt('datingTestSet2.txt')
#print('I am a: ', a[2:])

b, c = kNN.myfile2matrix('datingTestSet2.txt')
print('I am b: ', b)
print('I am c: ', c)

d, e = kNN.file2matrix('datingTestSet2.txt')
print('d: ', d)
print('e', e)
group, labels = kNN.createDataSet()
print("I'm group: ", group)
print("I'm labels: ", labels)

k = kNN.classify0([0, 0], group, labels, 3)
print('I am k: ', k)

print('b.min: ', b.min(0))
print('b.max: ', b.max(0))
print('b.mean: ', b.mean(0))
b_m = (b - b.mean(0))/(b.std(0))
print('b_m: ', b_m)

m = b.shape[0]
x = (b.max(0) - b.min(0))
b_a = (b - np.tile(b.min(0), (m, 1)))/np.tile(x, (m, 1))
print('b_a: ', b_a)

n, o, p = kNN.autoNorm(b)
print('n', n)
print('o: ', o)
print('p: ', p)
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.scatter(b[:, 0], b[:, 1], 15*c, 15*c)
#plt.show()

kNN.handwritingClassTest()
