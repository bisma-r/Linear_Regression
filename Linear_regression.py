def cost_function(w1, b1, m1, xy2):
    j1 = (1 / (2 * m1)) * sum_function(w1, b1, xy2, 0)
    return j1

def sum_function(w2, b2, xy1: list, d1):
    s = 0
    if d1 == 0:
        for i in xy1:
            s += ((w2 * i[0] + b2 - i[1]) ** 2)
    elif d1 == 1:
        for i in xy1:
            s += ((w2 * i[0] + b2 - i[1])* i[0])
    else:
        for i in xy1:
            s += (w2 * i[0] + b2 - i[1])
    return s

def update_para(w3, b3, a1, m2, xy3):
    old_w, old_b = w3, b3
    d2 = 1
    w3 = w3 - (a1 * ((1 / m2) * sum_function(old_w, old_b, xy3, d2)))
    d2 = 2
    b3 = b3 - (a1 * ((1/m2) * sum_function(old_w, old_b, xy3, d2)))
    return w3, b3

xy = [[1, 3], [2, 5], [3, 7], [4, 9], [5, 11]]

def run(l):
    w = 1
    b = 1
    m = len(l)
    a = 0.01
    j = 2
    max_iter = 10000 #chatGPT recommendation
    count = 0
    prev_j = float("inf")

    while abs(prev_j - j) > 1 and count < max_iter:
        prev_j = j
        j = cost_function(w, b, m, l)
        w, b = update_para(w, b, a, m, l)
        count +=1
        print ("w = ", w, " b = ", b, " j = ", j )
    print("No. of iterations: ", count)
    return w, b

#Predict salaries based on years of experience.

# Small dataset
data = {
    'YearsExperience': [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7,
                        3.9, 4.0, 4.0, 4.1, 4.5, 4.9, 5.1, 5.3, 5.9, 6.0,
                        6.8, 7.1, 7.9, 8.2, 8.7, 9.0, 9.5, 9.6, 10.3, 10.5],
    'Salary': [39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189,
               63218, 55794, 56957, 57081, 61111, 67938, 66029, 83088, 81363, 93940,
               91738, 98273, 101302, 113812, 109431, 105582, 116969, 112635, 122391, 121872]
}

data_list = [[x, y] for x, y in zip(data['YearsExperience'], data['Salary'])]

w, b = run(data_list)

def predict(val, w4, b4):
    ans = (val * w4) + b4
    print(ans)

predict(5.1, w, b)
data_dict = {x: y for x, y in data_list}
print(data_dict[5.1])