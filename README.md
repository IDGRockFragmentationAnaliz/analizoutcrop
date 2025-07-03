# outcropanaliz

Ссылка на данные [[данные]](https://disk.yandex.ru/d/uvB43D7ybMJPjA)

## Необходимые библиотеки

```commandline
pip install numpy
pip install pathlib
pip install matplotlib
pip install opencv-python
pip install scipy
pip install pyshp
pip install pandas
```

# Описание результата

- $s=\{s_i\}_{i=0}^{N}$ массив частиц
- $ b=\{b_0,...,b_n\} $ — разбиение на бины
- $ N_i $ — число частиц в $ i $-м бине,
- $ \Delta s_i = s_{i+1} - s_i $ — ширина $ i $-го бина,
- $ S = \sum_j S_j $ — сумма всех размеров частиц (или другая нормировочная величина),
- $ s_i^{\text{mid}} = \frac{s_i + s_{i+1}}{2} $ — середина $ i $-го бина,
- $ \rho(s_i^{\text{mid}}) $ — значение дифференциальной плотности в точке $ s_i^{\text{mid}} $.

---

Дифференциальная плотность:
$$
\rho(s_i^{\text{mid}}) = \frac{N_i}{\Delta s_i \cdot S}
$$