markdown_text_opgave1 = r"""
## Opgave 1.

Numerisk løsning af bølgeligningen og sammenligning med analytisk løsning. 


$\frac{\partial^2 u}{\partial x^2} - \frac{1}{v^2} \frac{\partial^2 u}{\partial t^2} = 0$, Med begyndelsesbetingelse: $u(x, 0) = f(x) = \exp\left(-\frac{x^2}{w^2}\right)\left(\frac{1}{2}e^{ik_0x} + \frac{1}{2}e^{-ik_0x}\right)$

"""


markdown_text_opgave2 = r"""
## Opgave 2.


$$\frac{\partial^2 u}{\partial x^2} - \frac{1}{v^2} \frac{\partial^2 u}{\partial t^2} = 0$$
Med begyndelsesbetingelse: 

$
u(x, 0) = \exp\left(-\frac{x^2}{w^2}\right) = f(x)
$
  og  
$
\frac{\partial u(x, 0)}{\partial t} = -v f'(x) = \frac{2x}{w^2} v \exp\left(-\frac{x^2}{w^2}\right)
$

"""


markdown_text_opgave3 = r"""
## Opgave 3.


$
\frac{\partial^2 u}{\partial x^2} - \frac{1}{v^2} \frac{\partial^2 u}{\partial t^2} - u = 0
$
Med begyndelsesbetingelse: 

$
u(x, 0) = \exp\left(-\frac{x^2}{w^2}\right)\left(\frac{1}{2}e^{ik_0x} + \frac{1}{2}e^{-ik_0x}\right)
$
  og  
$
\frac{\partial u(x, 0)}{\partial t} = 0
$

"""


markdown_text_opgave4 = r"""
## Opgave 4. Lineære del af den ikke lineære Schrödinger ligning (Fra projektplanen). 


$
i \frac{\partial u}{\partial t} + \frac{\partial^2 u}{\partial x^2} = 0
$
Med begyndelsesbetingelse: 
$
u(x, 0) = A \exp\left(-\left(\frac{x}{w}\right)^2\right), A=1
$

Analytisk løsing: 

$
u(x,t) = \frac{e^{-\frac{x^2}{4i t + w^2}}A w}{\sqrt{4i t + w^2}}, A=1
$


"""


markdown_text_projektplan_opgave1 = r"""
## Den Simple bølgeligning


$
\frac{\partial^2 u}{\partial t^2} - \frac{\partial^2 u}{\partial x^2} + u = 0
$
Med begyndelsesbetingelse: 

$
u(x, 0) = A \exp\left(-\left(\frac{x}{w}\right)^2\right),   A=0.5 
$


"""


markdown_text_projektplan_opgave2 = r"""
## Den Simple bølgeligning


$
\frac{\partial^2 u}{\partial t^2} - \frac{\partial^2 u}{\partial x^2} + u = 0
$
Med begyndelsesbetingelse: 

$
u(x, 0) = A \exp\left(-\left(\frac{x}{w}\right)^2\right) cos(5x),   A=0.5 
$


"""
