import numpy as np
import matplotlib.pyplot as plt

def linesearch(f, x, g, d):

    """
    Busca linear para determinar o passo alpha que satisfaz a condição de Armijo.
    
    Parâmetros:
        f: Função que está sendo minimizada.
        x: Ponto atual.
        g: Gradiente da função f no ponto x.
        d: Direção de busca.
        tau: Parâmetro de Armijo.
        gamma: Fator de redução do passo.
    
    Retorna:
        alpha: Tamanho de passo satisfazendo a condição de Armijo.
    """

    alpha = 1
    tau = 1e-3 
    gamma = 0.5

    while f(x + alpha * d) > f(x) + tau * alpha * g.dot(d):
        alpha *= gamma
    
    return alpha

def fin_diff(f, x, degree, h=1e-6):

    """
    Aproxima gradientes e derivadas de uma função em um ponto usando diferenças finitas.
    
    Parâmetros:
        f: A função a ser derivada. Deve aceitar um numpy array como entrada.
        x: Ponto onde se deseja calcular as derivadas (n-dimensional).
        degree: Grau da derivada (1 para gradiente, 2 para Hessiana).
        h: Passo usado nas diferenças finitas.
    
    Retorna:
        np.array: Vetor gradiente (se degree == 1) ou matriz Hessiana (se degree == 2).
    """

    n = len(x)
    
    if degree == 1:
        grad = np.zeros(n)
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1
            grad[j] = (f(x + h * e_j) - f(x - h * e_j)) / (2 * h)
        return grad
    
    elif degree == 2:
        hess = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                e_i = np.zeros(n)
                e_j = np.zeros(n)
                e_i[i] = 1
                e_j[j] = 1
                
                if i == j:
                    hess[i, j] = (f(x + h * e_i) - 2 * f(x) + f(x - h * e_i)) / (h ** 2)
                else:
                    hess[i, j] = (f(x + h * e_i + h * e_j) - f(x + h * e_i - h * e_j)- f(x - h * e_i + h * e_j) + f(x - h * e_i - h * e_j)) / (4 * h ** 2)
                    hess[j, i] = hess[i, j]
        return hess
    
    else:
        raise ValueError("degree deve ser 1 (gradiente) ou 2 (Hessiana).")

def gd(f, x0, grad, eps = 1e-5, alpha = 0.1, itmax = 10000, fd = False, h = 1e-7, plot = False, search = False):

    """
    Método do gradiente descendente com opções de busca linear, gradiente por diferenças finitas e plotagem.

    Parâmetros:
        f: Função a ser minimizada.
        x0: Ponto inicial como numpy array.
        grad: Função que calcula o gradiente (usada se fd=False).
        eps: Tolerância para o critério de parada.
        alpha: Tamanho do passo.
        itmax: Número máximo de iterações permitidas.
        fd: True para calcular o gradiente por diferenças finitas.
        h: Passo usado para diferenças finitas.
        plot: True para plotar a trajetória (válido apenas para R²).
        search: True para realizar busca linear.
    
    Retorna:
        x: Ponto final encontrado.
        k: Número de iterações realizadas.
    """

    x = x0.copy()
    k = 0
    trajectory = [x.copy()]
    
    if fd:
        while np.linalg.norm(fin_diff(f, x, 1, h)) > eps and k < itmax:
        
            k += 1
            g = fin_diff(f, x, 1, h)
            d = -g

            if search:
                alpha = linesearch(f, x, g, d)

            x = x + alpha * d
            trajectory.append(x.copy())
    else:
        while np.linalg.norm(grad(x)) > eps and k < itmax:
        
            k += 1
            g = grad(x)
            d = -g

            if search:
                alpha = linesearch(f, x, g, d)

            x = x + alpha * d
            trajectory.append(x.copy())

    if plot and len(x0) == 2:
        trajectory = np.array(trajectory)
        x_vals = np.linspace(min(trajectory[:, 0]) - 1, max(trajectory[:, 0]) + 1, 100)
        y_vals = np.linspace(min(trajectory[:, 1]) - 1, max(trajectory[:, 1]) + 1, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.array([[f(np.array([xx, yy])) for xx in x_vals] for yy in y_vals])

        plt.figure(figsize=(8, 6))
        plt.contour(X, Y, Z, levels=50, cmap="viridis")
        plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='red')
        plt.show()

    return x, k

def newton(f, x0, grad, hess, eps = 1e-5, alpha = 0.1, itmax = 10000, fd = False, h = 1e-7, plot = False, search = False):

    """
    Método de Newton para otimização não-linear.
    """

    x = x0.copy()
    k = 0
    g = fin_diff(f, x, 1, h) if fd else grad(x)
    trajectory = [x.copy()]
    I = np.eye(len(x))
    
    while np.linalg.norm(g) > eps and k < itmax:
        
        k += 1
        H = fin_diff(f, x, 2, h) if fd else hess(x)
        
        try:
            np.linalg.cholesky(H)
        except np.linalg.LinAlgError:
            H = 0.9 * H + 0.1 * I
        
        d = np.linalg.solve(H, -g)
        
        while d @ g > -1e-3 * np.linalg.norm(g) * np.linalg.norm(d):
            H = 0.9 * H + 0.1 * I
            d = np.linalg.solve(H, -g)

        if search:
            alpha = linesearch(f, x, g, d)
        
        x = x + alpha * d
        g = fin_diff(f, x, 1, h) if fd else grad(x)
        trajectory.append(x.copy())

    if plot and len(x0) == 2:
        trajectory = np.array(trajectory)
        x_vals = np.linspace(min(trajectory[:, 0]) - 1, max(trajectory[:, 0]) + 1, 100)
        y_vals = np.linspace(min(trajectory[:, 1]) - 1, max(trajectory[:, 1]) + 1, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.array([[f(np.array([xx, yy])) for xx in x_vals] for yy in y_vals])

        plt.figure(figsize=(8, 6))
        plt.contour(X, Y, Z, levels=50, cmap="viridis")
        plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='red')
        plt.show()

    return x, k

def bfgs(f, x0, grad, eps=1e-5, alpha=0.1, itmax=10000, fd=False, h=1e-7, plot=False, search=False):

    """
    Método BFGS para minimizar uma função.
    """
    
    x = x0.copy()
    k = 0
    g = fin_diff(f, x, 1, h) if fd else grad(x)
    I = np.eye(len(x))
    H = I
    trajectory = [x.copy()]

    while np.linalg.norm(g) > eps and k < itmax:
        
        k += 1
        y = g
        s = x
        d = -H @ g
        
        while d @ g > -1e-3 * np.linalg.norm(g) * np.linalg.norm(d):
            H = 0.9 * H + 0.1 * I
            d = -H @ g
        
        if search:
            alpha = linesearch(f, x, g, d)
        
        x = x + alpha * d
        g = fin_diff(f, x, 1, h) if fd else grad(x)
        y = g - y
        s = x - s

        denominator = s @ y
        if denominator > 1e-8:
            rho = 1.0 / denominator
            term1 = (1 + (y @ H @ y) * rho) * np.outer(s, s) * rho
            term2 = np.outer(H @ y, s) * rho
            term3 = np.outer(s, H @ y) * rho
            H = H + term1 - term2 - term3

        trajectory.append(x.copy())

    if plot and len(x0) == 2:
        trajectory = np.array(trajectory)
        x_vals = np.linspace(min(trajectory[:, 0]) - 1, max(trajectory[:, 0]) + 1, 100)
        y_vals = np.linspace(min(trajectory[:, 1]) - 1, max(trajectory[:, 1]) + 1, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.array([[f(np.array([xx, yy])) for xx in x_vals] for yy in y_vals])

        plt.figure(figsize=(8, 6))
        plt.contour(X, Y, Z, levels=50, cmap="viridis")
        plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='red')
        plt.show()

    return x, k

#A seguir três exemplos de funções a serem minimizadas:

#Função de Rosenbrock simplificada (quadrática).
#Essa função é uma parábola em R², com mínimo global no ponto (0,0).
def f1(x):
    return x[0]**2 + 2 * x[1]**2

def grad_f1(x):
    return np.array([2 * x[0], 4 * x[1]])

#Função de Rosenbrock (não quadrática).
#Um clássico problema de otimização, com mínimo global no ponto (1,1).
def f2(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def grad_f2(x):
    return np.array([
        -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2),
        200 * (x[1] - x[0]**2)
    ])

#Função de Himmelblau (não quadrática).
#Essa função tem quatro mínimos locais em: (3,2), (-2.805118, 3.131312), (-3.779310, -3.283186) e (3.584428, -1.848126).
def f3(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def grad_f3(x):
    return np.array([
        4 * x[0] * (x[0]**2 + x[1] - 11) + 2 * (x[0] + x[1]**2 - 7),
        2 * (x[0]**2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1]**2 - 7)
    ])

x,k = gd(f2,np.array([0,0]),grad_f2,plot = True)
print(f"x = {x}")
print(f"k = {k}")
