import math
import random as rd

xs = list(range(1, 11))
ys = [2**x for x in xs]
zs = [math.log2(y) for y in ys]

w, b = 0.0, 0.0
lr = 0.001

for epoch in range (5001):
    total_loss = 0
    for x, z in zip (xs, zs):
        pred_z = w * x + b
        err = pred_z - z

        total_loss += 0.5 * err**2

        w -= lr * err * x
        b -= lr * err

    if not (epoch%1000):
        print(f"At epoch {epoch}, model weight: {w:.6f}, bias: {b:.6f}, loss: {total_loss/len(xs)}")


print ("\n\nRandomized Inference Section")
inf_data = [rd.randint(32, 64) for _ in range(5)]
for bits in inf_data:
    print(f"A {bits} bit OS can address {(2**(w*bits+b))/1e9:0.6f} GB of memory.")

print ("\n\nUser Inference Section")
userInput = int(input("Enter the number of bits in your OS: "))
print(f"Your {userInput} bit OS can address {(2**(w*userInput+b))/1e9:0.6f} GB of memory.\n\n")