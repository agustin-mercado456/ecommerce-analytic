# ecommerce-analytic




## 🚀 Instalación y Configuración

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

### 📦 Dependencias incluidas

El archivo `environment.yml` incluye las siguientes librerías principales:
- **numpy**: Computación numérica
- **pandas**: Manipulación y análisis de datos
- **matplotlib**: Visualizaciones básicas
- **scikit-learn**: Machine Learning
- **scipy**: Funciones científicas
- **seaborn**: Visualizaciones estadísticas
- **mlflow**: Experimentación y tracking de ML

## 🔧 Solución de problemas comunes

### Error: "conda command not found"
- Asegúrate de que Anaconda/Miniconda esté instalado correctamente
- Reinicia tu terminal después de la instalación

### Error: "Environment already exists"
- Si el entorno ya existe, puedes actualizarlo:
  ```bash
  conda env update -f environment.yml
  ```

### Error: "Package conflicts"
- Elimina el entorno existente y créalo de nuevo:
  ```bash
  conda env remove -n ecommerce
  conda env create -f environment.yml
  ```


