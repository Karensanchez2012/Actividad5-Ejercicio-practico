Actividad 5 - Laboratorios 3.4 y 3.5 de Amazon SageMaker

Estudiante: Karen Dayanna Sanchez Pastuso

Curso: Ingeniería de Software, Universidad de La Salle

Rama de trabajo: dev_ksanchez90

¿Qué hice en esta actividad?

En esta actividad hice, el recorrido completo de un modelo de clasificación en Amazon SageMaker. Primero preparé los datos y entrené un modelo con XGBoost (laboratorio 3.4). Después lo desplegué, hice predicciones y eliminé los recursos para no gastar créditos (laboratorio 3.5). En cada laboratorio fui explicando qué hacía y por qué.

En el repositorio está lo siguiente:
- 3_4-machinelearning.ipynb: el notebook del laboratorio 3.4, con mi nombre y la explicación de cada paso.
- 3_5-machinelearning.ipynb: el notebook del laboratorio 3.5, con mi nombre, la explicación de cada paso y la validación del endpoint.
- evidencias: las capturas de pantalla completa de ambos laboratorios, que subí desde JupyterLab en AWS (de la 1 a la 19 son del laboratorio 3.4 y las que empiezan por 35_ son del 3.5).

Cómo preparé el entorno en AWS

Cada laboratorio abre una cuenta temporal de AWS y todo se borra cuando termina, así que repetí esta preparación en cada uno y subí mi trabajo a GitHub antes de finalizar.

Primero inicié el laboratorio en AWS Academy, abrí SageMaker y esperé a que mi instancia MyNotebook pasara de Pending a InService para entrar a JupyterLab. Luego, en la terminal, cloné el repositorio, entré a la carpeta y me cambié a mi rama con git checkout dev_ksanchez90, para no trabajar sobre main.

Configuré mi nombre y mi correo institucional en git para que los commits quedaran a mi nombre. Copié el notebook del laboratorio dentro del repositorio, creé la carpeta evidencias y excluí del control de versiones el archivo donde guardé mis credenciales de GitHub, para no subirlas por error. También escribí mi nombre en la primera celda de cada notebook. Los commits los hice desde la terminal de AWS con git add, git commit y git push, y al terminar borré mi token de acceso.

Laboratorio 3.4: formación de un modelo

El caso es el de un proveedor de salud que quiere detectar anomalías en pacientes ortopédicos. Cada paciente tiene seis medidas biomecánicas y una clase que dice si es normal o anormal.

Empecé importando los datos. Instalé scipy y scikit-learn, importé las librerías y descargué el Vertebral Column Dataset. Usé la versión de dos clases (column_2C_weka.arff) porque mi problema es de clasificación binaria, y la pasé a un DataFrame de pandas. Como los algoritmos trabajan con números, cambié las etiquetas de texto por 1 para Abnormal y 0 para Normal.

Después exploré los datos. Con df.shape vi que tenía 310 filas y 7 columnas, y con df.columns reconocí las seis características (pelvic_incidence, pelvic_tilt, lumbar_lordosis_angle, sacral_slope, pelvic_radius y degree_spondylolisthesis) y la columna class. Moví class a la primera posición porque XGBoost en SageMaker exige que el valor objetivo vaya primero.

Luego dividí los datos con train_test_split. Separé 80% para entrenamiento y 20% para prueba y validación, y ese 20% lo partí por la mitad. Me quedaron 248 filas de entrenamiento, 31 de prueba y 31 de validación. Usé stratify porque tengo pocos datos y quería que cada conjunto tuviera una proporción parecida de cada clase, y random_state=42 para que la división se pueda repetir. Lo comprobé con value_counts: en entrenamiento hay 168 casos anormales y 80 normales, y en prueba y validación hay 21 anormales y 10 normales cada uno. El conjunto de entrenamiento sirve para aprender, el de validación para ir revisando durante el entrenamiento y el de prueba lo guardo para evaluar el modelo al final.

Después subí los datos a Amazon S3, porque los trabajos de entrenamiento de SageMaker leen desde ahí y no desde el notebook. Definí el bucket, el prefijo lab3 y los nombres de los archivos, y escribí una función que convierte cada DataFrame a CSV en memoria, sin encabezados ni índice, y lo sube al bucket. Los guardé sin encabezados ni índice porque XGBoost solo espera los datos.

Para entrenar obtuve la imagen de XGBoost (versión 1.0-1) que ya trae SageMaker y definí los hiperparámetros: num_round en 42, eval_metric auc y objective binary:logistic. Elegí binary:logistic porque hay dos clases y así el modelo devuelve una probabilidad entre 0 y 1, y auc porque mide qué tan bien separa las dos clases. Son valores iniciales para un primer intento.

Configuré el entrenador con ModelTrainer: la imagen del contenedor, el rol de ejecución, una sola instancia ml.m5.4xlarge (el conjunto es pequeño y no necesito más), la ruta de salida en S3 y los hiperparámetros. Definí dos canales de datos, train y validation, indicando su ruta en S3 y el formato text/csv. Por último ejecuté xgb_model.train y, en la consola de SageMaker, en Trabajos de entrenamiento, comprobé que el trabajo terminó en estado Completed.

Laboratorio 3.5: implementación de un modelo

En este laboratorio el notebook primero deja listo el modelo. Repite los pasos de preparación (cargar datos, dividir, subir a S3, contenedor e hiperparámetros), crea el estimator con una instancia ml.m4.xlarge y ejecuta fit con los canales de entrenamiento y validación. Termina con el mensaje ready for hosting!, que significa que el modelo ya está entrenado y listo para alojarse.

En el primer paso desplegué el modelo con xgb_model.deploy en una instancia ml.m4.xlarge y con serializador CSV, y obtuve el predictor xgb_predictor. SageMaker crea el modelo, la configuración del endpoint y el endpoint, que es lo que permite pedirle predicciones al modelo en tiempo real.

En el segundo paso hice predicciones. Revisé el conjunto de prueba (31 filas y 7 columnas) y tomé la primera fila sin la columna class, porque el predictor no necesita la clase. La convertí a texto CSV y se la envié con predict. El resultado fue 0.9966, que no es un 0 o un 1 sino una probabilidad. Como esa fila realmente es de clase 1, la predicción fue correcta.

En el tercer paso eliminé el endpoint con delete_endpoint, junto con su configuración. Un endpoint activo sigue gastando créditos mientras exista, por eso hay que borrarlo cuando ya no se usa.

En el cuarto paso hice una transformación por lotes (batch transform), porque mandar las filas una por una es lento. Tomé todas las filas de prueba sin la clase, las subí a S3 como batch-in.csv y configuré un transformador con una instancia ml.m4.xlarge. SageMaker levanta una instancia con el modelo, predice todas las filas, guarda los resultados en S3 y apaga la instancia, sin dejar un endpoint prendido. Luego descargué el resultado a un DataFrame (target_predicted) y, con la función binary_convert, pasé cada probabilidad a 0 o 1 usando un umbral de 0.65. Al comparar con los valores reales, en las primeras 10 filas el modelo dijo que todas eran anormales: acertó las 8 que realmente lo son y se equivocó en 2 que son normales (las filas 230 y 297). Es un modelo inicial sin ajustar, y se va a evaluar con métricas y mejorar en los siguientes laboratorios.

Validación con xgb_predictor y lectura del resultado

Como el notebook ya había borrado su propio endpoint en el tercer paso, para la validación me conecté con Predictor al endpoint que había quedado activo (estado InService en la consola) y lo llamé xgb_predictor, con serializador CSV. Después de la prueba lo eliminé también y comprobé en la consola que ya no quedaban puntos de conexión.

La entrada fue "63.0,12.5,45.0,50.5,125.0,3.0", que corresponde en orden a la incidencia pélvica, la inclinación pélvica, el ángulo de lordosis lumbar, la pendiente sacra, el radio pélvico y el grado de espondilolistesis. El endpoint devolvió 0.0169.

Como el modelo usa binary:logistic, ese número es la probabilidad de que el paciente sea de la clase 1, es decir, Anormal. Un 0.0169 es cerca de 1.7%, muy por debajo del umbral de 0.65 que usé en binary_convert (y también de 0.5), así que el paciente se clasifica como Normal, con una probabilidad de normalidad de más o menos 98%. En palabras simples, con esas medidas el modelo no ve señales de hernia discal ni de espondilolistesis. Hay que recordar que es un modelo inicial sin ajustar y que su resultado es solo un apoyo, no reemplaza el diagnóstico de un especialista.

Commits

Realice 3 commits por laboratorio, todos desde la terminal del notebook en AWS.
