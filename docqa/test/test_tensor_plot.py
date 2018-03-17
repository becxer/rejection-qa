#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

q = "Who got the ball?"
c = ["The boy","got","the","ball","from","play","ground","!"]
gt = "The boy"
s,e = (3, 4)
pred = "play ground"

start_dist = np.array([
    1,2,3,4,3,2,1,1,
    1,4,4,1,1,3,3,1,
    5,4,3,2,1,2,3,4,
    1,1,2,1,1,1,2,1])

end_dist = np.array([
    5,4,3,2,1,2,3,4,
    1,4,4,1,1,3,3,1,
    1,2,3,4,3,2,1,1,
    1,1,2,1,1,1,2,1])

c_interval = np.arange(0.0, start_dist.shape[0], 1)
c_label = c
plt.figure(1)
plt.subplot(211)
plt.plot(c_interval, start_dist, color='r')
plt.title("Q : " + q + " // A : " + gt)
plt.text(0, 40, r'Predict : %s [%d:%d]' %(pred, s,e) , color='b')
axes = plt.gca()
axes.set_ylim([0 , 50])

plt.subplot(212)
plt.bar(c_interval, end_dist, color='g')
plt.xticks(c_interval, c_label, rotation=90)
axes = plt.gca()
axes.set_ylim([0 , 50])

plt.show()
#plt.savefig('dist')
