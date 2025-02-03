# Optimization-Methods
Este repositório contém implementações de métodos de otimização numérica, incluindo:

- Gradiente Descendente
- Método de Newton
- Método BFGS
- Busca linear (Armijo)
- Cálculo de gradientes e Hessianas via diferenças finitas

As implementações são feitas em Python utilizando NumPy e Matplotlib para visualização.

### 📌 Funcionalidades

- Suporte para gradiente analítico e aproximação por diferenças finitas.
- Opção de busca linear para escolha adaptativa do tamanho do passo.
- Visualização da trajetória de otimização em funções bidimensionais.
- Implementação modular, permitindo fácil extensão para novos métodos.

### 🚀 Aplicações em Machine Learning
A otimização numérica desempenha um papel fundamental no treinamento de modelos de Machine Learning. Os métodos implementados neste repositório podem ser usados para:

- Treinamento de Redes Neurais: O Gradiente Descendente é a base para algoritmos como SGD, Adam e RMSprop, usados no ajuste dos pesos de redes neurais profundas.
- Regressão Logística e Linear: O método de Newton pode acelerar a convergência nesses modelos convexos.
- SVMs (Máquinas de Vetores de Suporte): Algoritmos de otimização como BFGS podem ser utilizados para minimizar funções de perda convexas.
- Ajuste de Hiperparâmetros: Métodos de otimização podem ser empregados para encontrar hiperparâmetros ótimos em modelos de ML.

### 💡 Como Usar
Exemplo: Otimização da Função de Rosenbrock
```python
import numpy as np
from optimization_methods import gd, f2, grad_f2  

x0 = np.array([0, 0])  # Ponto inicial  
x_opt, iters = gd(f2, x0, grad_f2, plot=True)  

print(f"Resultado: {x_opt}")  
print(f"Iterações: {iters}")
```
Exemplo: Aplicação na Minimização de Erro em Regressão
```python
import numpy as np
from optimization_methods import gd, fin_diff  

# Função de erro quadrático para regressão linear  
def mse_loss(w):  
    X = np.array([[1, 1], [1, 2], [1, 3]])  # Dados  
    y = np.array([2, 2.5, 3.5])  # Rótulos  
    pred = X @ w  
    return np.mean((pred - y) ** 2)  

x0 = np.array([0.0, 0.0])  # Inicialização dos pesos  
grad = lambda x: fin_diff(mse_loss, x, degree=1)  # Gradiente via diferenças finitas  

x_opt, iters = gd(mse_loss, x0, grad, plot=False)  
print(f"Pesos ótimos: {x_opt}")
```
