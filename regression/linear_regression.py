xs = [1, 2, 3, 4, 5]
ys = [8, 16, 24, 32, 40]

w = 0.0; b = 0.0
lr = 0.01

for epoch in range(2501):
    total_loss = 0
    for x, y in zip(xs, ys):
        pred_y = w * x + b
        err = pred_y - y

        total_loss += 0.5 * err**2

        w -= lr * err * x
        b -= lr * err

    if not (epoch % 500):
        print(f"At epoch {epoch}, model weight: {w:.6f}, bias: {b:.6f}, loss: {total_loss/len(xs)}")