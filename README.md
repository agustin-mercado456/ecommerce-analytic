# ecommerce-analytic




##  Instalación y Configuración

### Prerrequisitos
- Anaconda o Miniconda instalado en tu sistema

### Pasos para configurar el entorno

1. **Clonar el repositorio**
   ```bash
   git clone <url-del-repositorio>
   cd ecommerce-analytic
   ```

2. **Crear el entorno conda**
   ```bash
   conda env create -f environment.yml
   ```

3. **Activar el entorno**
   ```bash
   conda activate ecommerce
   ```

4. **Verificar la instalación**
   ```bash
   python --version
   conda list
   ```

###  Dependencias incluidas

El archivo `environment.yml` incluye las siguientes librerías principales:
- **numpy**: Computación numérica
- **pandas**: Manipulación y análisis de datos
- **matplotlib**: Visualizaciones básicas
- **scikit-learn**: Machine Learning
- **scipy**: Funciones científicas
- **seaborn**: Visualizaciones estadísticas
- **mlflow**: Experimentación y tracking de ML
- **optuna**: Búsqueda de hiperparámetros óptimos

## Recomendaciones de Ejecución

Para una correcta exploración del proyecto, se recomienda seguir el siguiente orden al ejecutar los notebooks:

- Ejecutar el notebook 01_EDA_and_Cleaning para realizar la limpieza y exploración inicial de los datos.

- Continuar con 02_Feature_Engineering_andRFM para generar las variables y aplicar la segmentación RFM.

- Finalmente, dirigirse a la carpeta K-means para analizar los resultados del clustering y las recomendaciones generadas.

### Conclusión general

El análisis completo y las recomendaciones finales se encuentran en el notebook conclusion_general.ipynb.


