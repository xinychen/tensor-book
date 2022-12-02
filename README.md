# 从线性代数到张量分解

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Python 3.7](https://img.shields.io/badge/Python-3.7-blue.svg)
[![GitHub stars](https://img.shields.io/github/stars/xinychen/tensor-book.svg?logo=github&label=Stars&logoColor=white)](https://github.com/xinychen/tensor-book)

<h6 align="center">Made by Xinyu Chen (陈新宇) • :globe_with_meridians: <a href="https://xinychen.github.io">https://xinychen.github.io</a></h6>

## 目录

- **Kronecker积与Kronecker分解**
  - Kronecker积定义
  - Kronecker积基本性质
  - Kronecker积特殊性质
  - 朴素Kronecker积分解
  - 广义Kronecker积分解
  - 模型参数压缩问题

## 作者申明

- 撰写本文的初衷在于传播知识，为感兴趣的读者提供参考素材。
- 禁止将本文放在其他网站上，唯一下载网址为[https://xinychen.github.io/books/tensor_book.pdf](https://xinychen.github.io/books/tensor_book.pdf)。
- 禁止将本文用于任何形式的商业活动。

<h2 align="center">代数结构</h2>
<p align="right"><a href="#从线性代数到张量分解"><sup>▴ 回到顶部</sup></a></p>

**例.** 写出张量的lateral切片与horizontal切片。

```python
import numpy as np

X = np.zeros((2, 2, 2))
X[:, :, 0] = np.array([[1, 2], [3, 4]])
X[:, :, 1] = np.array([[5, 6], [7, 8]])
print('lateral slices:')
print(X[:, 0, :])
print()
print(X[:, 1, :])
print()
print('horizonal slices:')
print(X[0, :, :])
print()
print(X[1, :, :])
print()
```

<h2 align="center">Kronecker分解与Kronecker分解</h2>
<p align="right"><a href="#从线性代数到张量分解"><sup>▴ 回到顶部</sup></a></p>

**例.** 计算Kronecker积并求Kronecker分解。

- 计算Kronecker积

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6, 7], [8, 9, 10]])
X = np.kron(A, B)
print('X = ')
print(X)
```

- 求Kronecker分解

```python
def permute(mat, A_dim1, A_dim2, B_dim1, B_dim2):
    ans = np.zeros((A_dim1 * A_dim2, B_dim1 * B_dim2))
    for j in range(A_dim2):
        for i in range(A_dim1):
            ans[A_dim1 * j + i, :] = mat[i * B_dim1 : (i + 1) * B_dim1,
                                         j * B_dim2 : (j + 1) * B_dim2].reshape(B_dim1 * B_dim2, order = 'F')
    return ans

X_tilde = permute(X, 2, 2, 2, 3)
print('X_tilde = ')
print(X_tilde)
print()
u, s, v = np.linalg.svd(X_tilde, full_matrices = False)
A_hat = np.sqrt(s[0]) * u[:, 0].reshape(2, 2, order = 'F')
B_hat = np.sqrt(s[0]) * v[0, :].reshape(2, 3, order = 'F')
print('A_hat = ')
print(A_hat)
print()
print('B_hat = ')
print(B_hat)
```

<h2 align="center">Kronecker分解与Kronecker分解</h2>
<p align="right"><a href="#从线性代数到张量分解"><sup>▴ 回到顶部</sup></a></p>

**例.** 计算张量与矩阵的模态积。

```python
import numpy as np

X = np.zeros((2, 2, 2))
X[:, :, 0] = np.array([[1, 2], [3, 4]])
X[:, :, 1] = np.array([[5, 6], [7, 8]])
A = np.array([[1, 2], [3, 4], [5, 6]])
Y = np.einsum('ijh, ki -> kjh', X, A)
print('frontal slices of Y:')
print(Y[:, :, 0])
print()
print(Y[:, :, 1])
print()
```
