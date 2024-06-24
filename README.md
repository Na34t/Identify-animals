# Clasificador de Animales
Este es un clasificador de imágenes que determina con una foto de prueba si un animal es uno de las clases definidas en proyecto.py

Como requisito recomendamos utilizar el navegador de Anaconda y desde el navegador abrir el IDE donde gustes ejecutar el proyecto:
Windows: https://docs.anaconda.com/free/anaconda/install/windows/

NOTA: A varios que teníamos windows nos presentaba problema al ejecutarlo pero con Anaconda el problema pareció solucionarse.

Para instalar las bibliotecas en tu entorno virtual utiliza el comando:
```
pip install -r requirements.txt
```
Para ejecutar el proyecto utiliza el comando:
```
python proyecto.py
```
# Experimentación y pruebas
En la carpeta Test agrega la imagen que quieras poner a prueba y reemplázala con el nombre de "test1", "test2","test3" o "test4"

En la función test_image_path:
```
test_image_paths = ['Test/test1.jpeg', 'Test/test2.jpeg', 'Test/test3.jpeg', 'Test/test4.jpeg']
for image_path in test_image_paths:
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f'Clase: {test(net, image_path)}')
    plt.show()
```
Modificar la extensión dependiendo de la imagen de prueba que utilices.
