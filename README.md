# 从线性代数到张量分解

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Python 3.7](https://img.shields.io/badge/Python-3.7-blue.svg)
[![GitHub stars](https://img.shields.io/github/stars/xinychen/tensor-book.svg?logo=github&label=Stars&logoColor=white)](https://github.com/xinychen/tensor-book)

<h6 align="center">Made by Xinyu Chen • :globe_with_meridians: <a href="https://xinychen.github.io">https://xinychen.github.io</a></h6>

**目录**

- Kronecker分解

<h2 align="center">Kronecker分解</h2>
<p align="right"><a href="#从线性代数到张量分解"><sup>▴ 回到顶部</sup></a></p>

【例】给定矩阵$\boldsymbol{A}=\begin{bmatrix} 1 & 2 \\ 3 & 4 \\ \end{bmatrix}$与$\boldsymbol{B}=\begin{bmatrix} 5 & 6 & 7 \\ 8 & 9 & 10 \\ \end{bmatrix}$，试写出两者之间的Kronecker积$\boldsymbol{X}=\boldsymbol{A}\otimes\boldsymbol{B}$，并求Kronecker分解$\displaystyle\hat{\boldsymbol{A}},\hat{\boldsymbol{B}}=\argmin_{\boldsymbol{A},\boldsymbol{B}}~\|\boldsymbol{X}-\boldsymbol{A}\otimes\boldsymbol{B}\|_{F}^{2}$。
