 
# Import required package
import numpy as np
 

# Compute Pr(S0 -> ... -> S5) by "Detection Policy"

# Taking a 3 * 3 matrix
A = np.array([[0.1, 0.03, 0.07, 0.8],
              [0, 0.1, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]) - np.eye(4)
b = np.array([[0],
              [0],
              [-1.0],
              [-0.1]])

print(b)
print(A)
# Calculating the inverse of the matrix
# print(np.linalg.inv(A))
print(np.linalg.solve(A, b))


# Compute Pr(S0 -> ... -> S5) for <= 5 time steps
A = np.array([[0.1, 0.03, 0.07, 0.8, 0, 0],
              [0, 0.1, 0, 0, 0.9, 0],
              [0, 0, 0, 0, 0, 1.0],
              [0, 0, 0, 0, 0.9, 0.1],
              [0, 0, 0, 0, 1.0, 0],
              [0, 0, 0, 0, 0, 1.0]])
A_trans = A.transpose()
print(A_trans)
state_prob = []
init = np.array([1, 0, 0, 0, 0, 0]).reshape(-1, 1)
state_prob.append(init)

print(np.reshape(state_prob[0], (1, -1)))
for i in range(5):
    state_prob.append(np.matmul(A_trans, state_prob[i]))
    print(np.reshape(state_prob[i+1], (1, -1)))

# find prob
prob_percentage = 100 * state_prob[5][5] / np.sum(state_prob[5]) # although the sum is definitely 1
print("print prob: ", prob_percentage)


state_prob[5][5] = state_prob[5][5] - 0.1
print("state_prob: ", state_prob)
cmp = 1
k = 0

temp1 = []
temp2 = []

while k <= 0.07:
    A_new = np.copy(A)
    A_new[0][2] -=  k
    A_new[0][3] +=  k
    A_new_trans = A_new.transpose()
    new_state_prob = np.matmul(A_new_trans, init)
    for i in range(4):
        new_state_prob = np.matmul(A_new_trans, new_state_prob)
    # print(new_state_prob)
    temp1.append(pow(new_state_prob[5] - state_prob[5][5], 2))
    temp2.append(new_state_prob[5]*100)
    if pow(new_state_prob[5] - state_prob[5][5], 2) < cmp:
        cmp = pow(new_state_prob[5] - state_prob[5][5], 2)
        candidate = k
    k += 0.0001
    
print(cmp, " & ", candidate)


import matplotlib.pyplot as plt

y = temp2
x = np.linspace(0.07, 0, 700)

plt.scatter(x, y)

plt.plot(x[0], y[0], marker='x', color='r', markersize=10)
plt.text(x[0]-0.002, y[0], f'({x[0]}, {"%.3f" % y[0].item()}%)', ha='left', va='bottom')
plt.plot(x[-1], y[-1], marker='x', color='r', markersize=10)
plt.text(x[-1]+0.002, y[-1]-0.2, f'({x[-1]}, {"%.3f" % y[-1].item()}%)', ha='right', va='bottom')

plt.xlim(max(x), min(x))
plt.xlabel("Pd02")
plt.ylabel("Probabiliy of collision (%)")
plt.title("Plot of Probabiliy of Collision - Pd02")

plt.show()



# # ----------

# # Compute Pr(S0 -> ... -> S5) by "Just Drive Policy"

# # Taking a 3 * 3 matrix
# A2 = np.array([[0.95, 0, 0, 0],
#               [0, 0.1, 0, 0],
#               [0, 0, 0, 0],
#               [0, 0, 0, 0]]) - np.eye(4)
# b2 = np.array([[-0.05],
#               [0],
#               [-1.0],
#               [-0.1]])

# print(b2)
# print(A2)
# # Calculating the inverse of the matrix
# # print(np.linalg.inv(A))
# print(np.linalg.solve(A2, b2))



# Compute Pr(S0 -> ... -> S5) for <= 5 time steps
A2 = np.array([[0.95, 0, 0, 0, 0, 0.05],
              [0, 0.1, 0, 0, 0.9, 0],
              [0, 0, 0, 0, 0, 1.0],
              [0, 0, 0, 0, 0.9, 0.1],
              [0, 0, 0, 0, 1.0, 0],
              [0, 0, 0, 0, 0, 1.0]])
A2_trans = A2.transpose()
print(A2_trans)
state_prob2 = []
state_prob2.append(np.array([1, 0, 0, 0, 0, 0]).reshape(-1, 1))
# state_prob.append(np.array([2, 0, 0, 0, 0, 0]).reshape(-1, 1))

print(np.reshape(state_prob2[0], (1, -1)))
for i in range(5):
    state_prob2.append(np.matmul(A2_trans, state_prob2[i]))
    print(np.reshape(state_prob2[i+1], (1, -1)))

# find prob
prob2_percentage = 100 * state_prob2[5][5] / np.sum(state_prob2[5]) # although the sum is definitely 1
print(prob2_percentage)

