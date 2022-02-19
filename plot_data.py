import pickle 
import matplotlib.pyplot as plt

readed = open('generations_data.obj', 'rb') 
generations_data = pickle.load(readed)

min_list = generations_data['min']
max_list = generations_data['max']
avg_list = generations_data['avg']
plt.plot(min_list, label = 'min')
plt.plot(max_list, label = "max")
plt.plot(avg_list, label = 'avg')
plt.xlabel('generation')
plt.ylabel('fitness')
plt.title('fitness/generation grapgh')
plt.legend()
plt.show()


