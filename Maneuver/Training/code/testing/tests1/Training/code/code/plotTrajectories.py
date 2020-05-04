import matplotlib.pyplot as plt
import csv	


with open('trajectory_position.csv', 'rU') as data:
    reader = csv.reader(data)
    for row in reader:
        for cell in row:
             cell=float(cell)
             print('cell is: ', cell)

        plt.plot(row)
#        plt.ylabel('position for each traj run')
        plt.show()


                 
        
