import torch
from torch.autograd import grad
import time
from typing import Callable

# Задание 1: Создание и манипуляции с тензорами


def task11() -> None:
    print("============== 1.1 Создание тензоров ==============")

    t1 = torch.randint(
        0, 2, size=(3, 4)
    )  # Тензор размером 3x4, заполненный случайными числами от 0 до 1
    t2 = torch.zeros((2, 3, 4))  # Тензор размером 2x3x4, заполненный нулями
    t3 = torch.ones((5, 5))  # Тензор размером 5x5, заполненный единицами
    t4 = torch.arange(0, 16).reshape(
        (4, 4)
    )  # Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)

    print(t1)
    print(t2)
    print(t3)
    print(t4)


def task12(a: torch.TensorType, b: torch.TensorType) -> None:
    print("============== 1.2 Операции с тензорами ==============")

    print(a.T)  # Транспонирование тензора A
    print(a @ b)  # Матричное умножение A и B
    print(a * b.T)  # Поэлементное умножение A и транспонированного B
    print(torch.sum(a))  # Вычислите сумму всех элементов тензора A


def task13() -> None:
    print("============== 1.3 Индексация и срезы ==============")
    a = torch.arange(125).reshape((5, 5, 5))

    center_layer_index = a.shape[0] // 2 if a.shape[0] % 2 == 0 else a.shape[0] // 2 + 1
    center_matrice_index = a.shape[1] // 2
    last_col_index = a.shape[1] - 1

    print(a[0])  # Первая строка
    print(a[:, last_col_index])  # Последний столбец
    print(
        a[
            center_layer_index,
            center_matrice_index - 1 : center_matrice_index + 1,
            center_matrice_index - 1 : center_matrice_index + 1,
        ]
    )  # Подматрицу размером 2x2 из центра тензора

    print(a[::2, ::2, ::2])  # Все элементы с четными индексами


def task14() -> None:
    print("============== 1.4 Работа с формами ==============")
    a = torch.arange(24)

    print(a.reshape(2, 12))
    print(a.reshape(3, 8))
    print(a.reshape(4, 6))
    print(a.reshape(2, 3, 4))
    print(a.reshape(2, 2, 2, 3))


# Тесты
# task11()
# task12(
#     torch.Tensor([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]),
#     torch.Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
# )
# task13()
# task14()


# Задание 2: Автоматическое дифференцирование


def task21() -> None:
    print("============== 2.1 Простые вычисления с градиентами ==============")
    x = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    y = torch.arange(4, 8, dtype=torch.float32).reshape(2, 2)
    z = torch.arange(8, 12, dtype=torch.float32).reshape(2, 2)

    # Включаем autograd для тензоров после reshape
    x.requires_grad_()
    y.requires_grad_()
    z.requires_grad_()

    f = (x**2 + y**2 + z**2 + 2 * x * y * z).sum()

    # Autograd: вычисление градиентов
    f.backward()
    print(f"x.grad = {x.grad}")
    print(f"y.grad = {y.grad}")
    print(f"z.grad = {z.grad}")

    # Вычисление градиентов аналитическим методом
    df_x = 2 * x + 2 * y * z
    df_y = 2 * y + 2 * x * z
    df_z = 2 * z + 2 * x * y

    # Проверка аналитического метода
    torch.testing.assert_close(x.grad, df_x)
    torch.testing.assert_close(y.grad, df_y)
    torch.testing.assert_close(z.grad, df_z)


def task22() -> None:
    print("============== 2.2 Градиент функции потерь ==============")
    x = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True
    )
    w = torch.randn(3, 1, requires_grad=True)
    b = torch.rand(3, 1, requires_grad=True)

    n = len(x)

    true_weights = torch.tensor([[0.2], [0.5], [0.3]])
    y_true = (x @ true_weights) + 1
    sigmoid = lambda x: 1 / (1 + torch.exp(-x))
    y_pred = sigmoid(x @ w + b)

    MSE = (1 / n) * torch.sum((y_pred - y_true) ** 2)

    # Autograd: вычисление градиентов
    MSE.backward()
    print(f"w.grad = {w.grad}")
    print(f"b.grad = {b.grad}")

    # Вычисление аналитически
    delta = y_pred * (1 - y_pred) * (y_pred - y_true)
    dw = (2 / n) * x.T @ delta
    db = (2 / n) * delta
    print(f"dw = {dw}")
    print(f"db = {db}")

    # Проверка
    torch.testing.assert_close(w.grad, dw)
    torch.testing.assert_close(b.grad, db)


def task23() -> None:
    print("============== 2.3 Цепное правило ==============")
    x = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True
    )

    f_x = torch.sin(x**2 + 1)
    dx = 2 * torch.cos(x**2 + 1) * x
    f_grad = grad(f_x, x, grad_outputs=torch.ones_like(f_x))[0]
    print(f"w.grad = {f_grad}")

    # Проверка
    torch.testing.assert_close(f_grad, dx)


# Тесты
# task21()
# task22()
# task23()


# Задание 3: Сравнение производительности CPU vs CUDA
def task3() -> None:
    # 3.1 Подготовка данных
    # Размеры отличаются от условия, чтобы выполнить матричное умножение
    # и другие операции
    x = torch.randn(200, 200, 200)
    y = torch.randn(200, 200, 200)
    z = torch.randn(200, 200, 200)

    print("Операция          | CPU (мс) | GPU (мс) | Ускорение")
    operations = (
        ("Матричное умножение", lambda vec: vec[0] @ vec[1] @ vec[2]),
        ("Поэлементное сложение", lambda vec: vec[0] + vec[1] + vec[2]),
        ("Поэлементное умножение", lambda vec: vec[0] * vec[1] * vec[2]),
        ("Транспонирование", lambda vec: vec[0].mT),
        ("Сумма элементов", lambda vec: torch.sum(vec[0])),
    )

    vec = (x, y, z)
    vec_gpu = tuple(v.cuda() for v in vec)  # заранее переносим тензоры на GPU

    for operation_name, operation in operations:
        time_operation(operation, vec, vec_gpu, operation_name)


def time_operation(
    operation: Callable,
    vec: tuple[torch.TensorType],
    vec_gpu: tuple[torch.TensorType],
    operation_name: str,
) -> None:
    """Функция для замера времени выполения операции"""
    # CPU
    cpu_start = time.perf_counter_ns()
    operation(vec)
    cpu_end = time.perf_counter_ns()

    # GPU
    gpu_start = torch.cuda.Event(enable_timing=True)
    gpu_end = torch.cuda.Event(enable_timing=True)

    gpu_start.record()
    operation(vec_gpu)
    gpu_end.record()

    # Ждёт окончания всех процессов, чтобы завершить работу
    torch.cuda.synchronize()

    # Рассчёт метрик
    cpu_time = (cpu_end - cpu_start) / 10**6
    gpu_time = gpu_start.elapsed_time(gpu_end)
    acceleration = cpu_time / gpu_time

    print(
        f"{operation_name} | {cpu_time:.1f}  |   {gpu_time:.1f}   |   {acceleration:.1f}"
    )


# Тесты
task3()
