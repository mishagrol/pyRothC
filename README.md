## pyRothC
____________
<p align="center">
    <em>Python version of The Rothamsted carbon model (RothC) 26.3.</em>
</p>
<p align="center">
<a href="https://github.com/tiangolo/fastapi/actions?query=workflow%3ATest+event%3Apush+branch%3Amaster" target="_blank">
    <img src="https://github.com/tiangolo/fastapi/workflows/Test/badge.svg?event=push&branch=master" alt="Test">
</a>
<a href="https://coverage-badge.samuelcolvin.workers.dev/redirect/tiangolo/fastapi" target="_blank">
    <img src="https://coverage-badge.samuelcolvin.workers.dev/tiangolo/fastapi.svg" alt="Coverage">
</a>
<a href="https://pypi.org/project/fastapi" target="_blank">
    <img src="https://img.shields.io/pypi/v/fastapi?color=%2334D058&label=pypi%20package" alt="Package version">
</a>
<a href="https://pypi.org/project/fastapi" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/fastapi.svg?color=%2334D058" alt="Supported Python versions">
</a>
</p>

---

**Documentation**: <a href="https://www.rothamsted.ac.uk/sites/default/files/RothC_guide_DOS.pdf" target="_blank">Rothamsted RothC Model</a>

**Source Code**: <a href="https://github.com/mishagrol/pyRothC" target="_blank">https://github.com/mishagrol/pyRothC</a>

---

pyRothc is a Python version of The Rothamsted carbon model (RothC) 26.3.


RothC is a model for the turnover of organic carbon in non-waterlogged topsoil that allows for the effects of soil type, temperature, soil moisture and plant cover on the turnover process.


## Requirements

Python 3.7+

SciPy

NumPy

Pandas

## Installation

<div class="termy">

```console
$ pip install pyRothC
```

</div>

## Example



```Python
from pyRothC import RothC

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

```


## Structure of the RothC model

**Credits**: <a href="https://www.bgc-jena.mpg.de/TEE/software/bgc-md/soil/Jenkinson1977SoilScience-S0003/Report.html" target="_blank">Theoretical Ecosystem Ecology group of the Max Planck Institute for Biogeochemistry</a>


<p align="center">
  <a href="https://fastapi.tiangolo.com"><img src="./plots/Logo.svg" alt="FastAPI"></a>
</p>

### Equations

$$
\begin{aligned}
& \frac{d \boldsymbol{C}}{\mathrm{d} t}=I\left(\begin{array}{c}
\gamma \\
1-\gamma \\
0 \\
0 \\
0
\end{array}\right) 
 +\left(\begin{array}{ccccc}
-k_1 & 0 & 0 & 0 & 0 \\
0 & -k_2 & 0 & 0 & 0 \\
a_{3,1} & a_{3,2} & -k_3+a_{3,3} & a_{3,4} & 0 \\
a_{4,1} & a_{4,2} & a_{4,3} & -k_4+a_{4,4} & 0 \\
0 & 0 & 0 & 0 & 0
\end{array}\right)\left(\begin{array}{l}
C_1 \\
C_2 \\
C_3 \\
C_4 \\
C_5
\end{array}\right) \\
&
\end{aligned}
$$
## Optional Dependencies

Matplotlib

## License

This project is licensed under the terms of the MIT license.