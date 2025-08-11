def cost_function(w1, b1, m1):
    j1 = (1 / (2 * m1)) * sum_function(w1, b1, xy, d)
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

def update_para(w3, b3, a1, m2):
    temp = w3
    d2 = 1
    w3 = w3 - (a1 * ((1 / m2) * sum_function(w3, b3, xy, d2)))
    d2 = 2
    b3 = b3 - (a1 * ((1/m2) * sum_function(temp, b3, xy, d2)))
    return w3, b3

xy = [[1, 3], [2, 5], [3, 7], [4, 9], [5, 11]]
w = 1
b = 1
m = len(xy)
a = 0.01
d = 0
j = 2
max_iter = 10000 #chatGPT recommendation
count = 0

while j > 1e-6 and count < max_iter:
    prevJ = j
    j = cost_function(w, b, m)
    w, b = update_para(w, b, a, m)
    count +=1
    print ("w = ", w, " b = ", b, " j = ", j )