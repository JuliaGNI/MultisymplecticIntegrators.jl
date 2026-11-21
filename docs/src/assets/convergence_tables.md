**CoefficientRestriction.** The `L²` error of the top trace against `u*` at `t = 1.0` on the joint ladder,
expected order `p + 1`, and the `L²` difference of consecutive runs on the self-difference
ladder, expected order `2p` in `h`. Run on commit `93be018` with Julia 1.13.4.

| ladder | p | D | expected | value 1 | value 2 | order 1–2 | value 3 | order 2–3 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| joint `nx = 20, 40, 80`, `h = 1/nx` | 1 | 1 | 2 | 2.3903e-03 | 5.9742e-04 | 2.000 | 1.4934e-04 | 2.000 |
| self-difference `h = 0.1, 0.05, 0.025, 0.0125`, `nx = 20` | 1 | 1 | 2 | 6.5866e-04 | 1.6514e-04 | 1.996 | 4.1316e-05 | 1.999 |
| joint `nx = 20, 40, 80`, `h = 1/nx` | 1 | 2 | 2 | 2.9928e-03 | 7.4806e-04 | 2.000 | 1.8701e-04 | 2.000 |
| self-difference `h = 0.1, 0.05, 0.025, 0.0125`, `nx = 20` | 1 | 2 | 2 | 7.6868e-04 | 1.9272e-04 | 1.996 | 4.8214e-05 | 1.999 |
| joint `nx = 20, 40, 80`, `h = 1/nx` | 2 | 1 | 3 | 4.6708e-05 | 5.6737e-06 | 3.041 | 7.0406e-07 | 3.011 |
| self-difference `h = 0.1, 0.05, 0.025, 0.0125`, `nx = 20` | 2 | 1 | 4 | 1.6699e-07 | 1.0459e-08 | 3.997 | 6.5406e-10 | 3.999 |
| joint `nx = 20, 40, 80`, `h = 1/nx` | 2 | 2 | 3 | 5.7019e-05 | 6.9258e-06 | 3.041 | 8.5942e-07 | 3.011 |
| self-difference `h = 0.1, 0.05, 0.025, 0.0125`, `nx = 20` | 2 | 2 | 4 | 1.9103e-07 | 1.1965e-08 | 3.997 | 7.4819e-10 | 3.999 |

**WeakInitialCondition.** The `L²` error of the top trace against `u*` at `t = 1.0` on the joint ladder,
expected order `p + 1`, and the `L²` difference of consecutive runs on the self-difference
ladder, expected order `2p` in `h`. Run on commit `93be018` with Julia 1.13.4.

| ladder | p | D | expected | value 1 | value 2 | order 1–2 | value 3 | order 2–3 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| joint `nx = 20, 40, 80`, `h = 1/nx` | 1 | 1 | 2 | 2.5720e-03 | 6.4298e-04 | 2.000 | 1.6074e-04 | 2.000 |
| self-difference `h = 0.1, 0.05, 0.025, 0.0125`, `nx = 20` | 1 | 1 | 2 | 6.6276e-04 | 1.6617e-04 | 1.996 | 4.1574e-05 | 1.999 |
| joint `nx = 20, 40, 80`, `h = 1/nx` | 1 | 2 | 2 | 3.2048e-03 | 8.0122e-04 | 2.000 | 2.0031e-04 | 2.000 |
| self-difference `h = 0.1, 0.05, 0.025, 0.0125`, `nx = 20` | 1 | 2 | 2 | 7.7326e-04 | 1.9387e-04 | 1.996 | 4.8503e-05 | 1.999 |
| joint `nx = 20, 40, 80`, `h = 1/nx` | 2 | 1 | 3 | 4.6711e-05 | 5.6738e-06 | 3.041 | 7.0406e-07 | 3.011 |
| self-difference `h = 0.1, 0.05, 0.025, 0.0125`, `nx = 20` | 2 | 1 | 4 | 1.6700e-07 | 1.0460e-08 | 3.997 | 6.5408e-10 | 3.999 |
| joint `nx = 20, 40, 80`, `h = 1/nx` | 2 | 2 | 3 | 5.7023e-05 | 6.9259e-06 | 3.041 | 8.5942e-07 | 3.011 |
| self-difference `h = 0.1, 0.05, 0.025, 0.0125`, `nx = 20` | 2 | 2 | 4 | 1.9103e-07 | 1.1965e-08 | 3.997 | 7.4821e-10 | 3.999 |

**WeakInitialAndBoundaryConditions.** The `L²` error of the top trace against `u*` at `t = 1.0` on the joint ladder,
expected order `p + 1`, and the `L²` difference of consecutive runs on the self-difference
ladder, expected order `2p` in `h`. Run on commit `93be018` with Julia 1.13.4.

| ladder | p | D | expected | value 1 | value 2 | order 1–2 | value 3 | order 2–3 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| joint `nx = 20, 40, 80`, `h = 1/nx` | 1 | 1 | 2 | 2.5720e-03 | 6.4298e-04 | 2.000 | 1.6074e-04 | 2.000 |
| self-difference `h = 0.1, 0.05, 0.025, 0.0125`, `nx = 20` | 1 | 1 | 2 | 6.6276e-04 | 1.6617e-04 | 1.996 | 4.1574e-05 | 1.999 |
| joint `nx = 20, 40, 80`, `h = 1/nx` | 1 | 2 | 2 | 3.2048e-03 | 8.0122e-04 | 2.000 | 2.0031e-04 | 2.000 |
| self-difference `h = 0.1, 0.05, 0.025, 0.0125`, `nx = 20` | 1 | 2 | 2 | 7.7326e-04 | 1.9387e-04 | 1.996 | 4.8503e-05 | 1.999 |
| joint `nx = 20, 40, 80`, `h = 1/nx` | 2 | 1 | 3 | 4.6711e-05 | 5.6738e-06 | 3.041 | 7.0406e-07 | 3.011 |
| self-difference `h = 0.1, 0.05, 0.025, 0.0125`, `nx = 20` | 2 | 1 | 4 | 1.6700e-07 | 1.0460e-08 | 3.997 | 6.5408e-10 | 3.999 |
| joint `nx = 20, 40, 80`, `h = 1/nx` | 2 | 2 | 3 | 5.7023e-05 | 6.9259e-06 | 3.041 | 8.5942e-07 | 3.011 |
| self-difference `h = 0.1, 0.05, 0.025, 0.0125`, `nx = 20` | 2 | 2 | 4 | 1.9103e-07 | 1.1965e-08 | 3.997 | 7.4821e-10 | 3.999 |

**The box scheme.** The `L²` error of the last level against `u*` at the grid nodes at
`t = 1.0` on the space ladder, and the `L²` difference of consecutive runs on the
self-difference ladder, expected order `2` in both. Run on commit `93be018` with Julia 1.13.4.

| ladder | D | expected | value 1 | value 2 | order 1–2 | value 3 | order 2–3 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| space `nx = 10, 20, 40`, `h = 1/640` | 1 | 2 | 4.8763e-03 | 1.1546e-03 | 2.078 | 2.8486e-04 | 2.019 |
| self-difference `h = 0.1, 0.05, 0.025, 0.0125`, `nx = 40` | 1 | 2 | 1.0629e-03 | 2.6724e-04 | 1.992 | 6.6906e-05 | 1.998 |
| space `nx = 10, 20, 40`, `h = 1/640` | 2 | 2 | 6.5394e-03 | 1.5499e-03 | 2.077 | 3.8231e-04 | 2.019 |
| self-difference `h = 0.1, 0.05, 0.025, 0.0125`, `nx = 40` | 2 | 2 | 1.2438e-03 | 3.1272e-04 | 1.992 | 7.8291e-05 | 1.998 |
