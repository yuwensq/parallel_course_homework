import matplotlib.pyplot as plt
import numpy as np
x = []
i = 4
while i < 256:
    x.append(i)
    i *= 2
x.append(i)
i += 128
while i <= 2048:
    x.append(i)
    i += 128


def draw_picture(file_name, title):
    y_thread = [[], [], []]
    with open(file_name, encoding='utf-8') as file:
        for j in range(3):
            y_thread[j] = list(map(float, file.readline().split()))
            y_thread[j] = three_degree_ni(y_thread[j])
    # l1, = plt.plot(x, y_thread[0], color='red', linewidth=1.0, linestyle='-.')
    l2, = plt.plot(x, y_thread[1], color='green', linewidth=1.0, linestyle='-.')
    l3, = plt.plot(x, y_thread[2], color='blue', linewidth=1.0, linestyle='-.')
    plt.title(title)
    plt.xlabel('scale')
    plt.ylabel('time/ms')
    plt.legend(handles=[l2, l3, ], labels=['row_distribute', 'col_distribute', ], loc='best')
    # plt.show()


def three_degree_ni(y):
    z1 = np.polyfit(x, y, 3)  # 用3次多项式拟合，输出系数从高到0
    p1 = np.poly1d(z1)  # 使用次数合成多项式
    return p1(x)


plt.subplot(221)
draw_picture("thread2.txt", "thread=2")
plt.subplot(222)
draw_picture("thread4.txt", "thread=4")
plt.subplot(223)
draw_picture("thread6.txt", "thread=6")
plt.subplot(224)
draw_picture("thread8.txt", "thread=8")
plt.show()
