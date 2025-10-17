Clase Hopfield:
Esta es la red en si, guarda los pesos y entrena con distintos métodos y recupera patrones. 

Metodos:
Train_hebb(): Entrena la red usando la regla de Hebb. Calcula la conexión sináptica entre las neuronas según los patrones y borra la diagonal para que una neurona no se conecte consigo misma. 
Train_pseudoinverse(): Entrena usando el método de la pseudoinversa de Moore-Penrose, mejora la estabilidad cuando hay patrones similares. 
Recall(): Intenta recuperar un patrón dado uno de entrada (puede poseer ruido). Usa actualización asíncrona, o sea, actualiza las neuronas una por una. Repite hasta que el patrón converge. Devuelve el patrón final recuperado.
Energy(): Calcula la energía del sistema. En una red hopfield la energía siempre baja hasta que llega a un minimio estable. Es una forma de ver si la red se está estabilizando. 
Calculate_overlap(): Mide cuanto se parece un patrón recuperado al original. Cuanto más cerca de 1 -> similares.

Funciones auxiliares:
Créate_reference_mark(): Crea una marca de referencia en la esquina superior izquierda.
Créate_circle(): Crea una figura circular en una matriz de 10x10. Los pixeles del borde del círculo se ponen ene 1, el resto en -1.
Add_noise(): agrega ruido al patrón, es decir, cambia aleatoriamente algunos pixeles de signo. No modifica la marca de referencia.
Print_pattern(): Muestra el patrón en consola usando. para los inactivos -1. Sirve para visualizar los patrones sin gráficos.
Find_center(): calcula el centro geométrico del patrón. Esto permite medir cuanto se desplazó la figura antes y después de recuperar el patrón.

Run_experiment(): Ejecuta un experimento completo:
1.	Muestra el patrón original.
2.	Le agrega ruido.
3.	Intenta recuperarlo con la red (recall).
4.	Muestra el resultado.
5.	Calcula:
•	El centro original y recuperado.
•	El error de posición.
•	El solapamiento con cada patrón aprendido.
•	La energía antes y después.
•	Cuántas iteraciones tardó en converger.
