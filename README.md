# Pytorch regression problem

### Технологии:
- Язык: &nbsp; `python` ;
- Фреймворки: &nbsp; `pytorch`, `fastapi` ;
- Библиотеки: &nbsp; `numpy`, `yaml` , `matplotlib` ;
- Frontend: &nbsp; `html`, `jinja` ;
- ML-алгоритмы: &nbsp; `ModelReg` .
  
### Описание:

<img width='400px' src='https://github.com/primera7790/pytorch_regression/blob/master/github_data/show_img.png' alt='square'/>

&nbsp; &nbsp; **Задача:** &nbsp; Определение координат центра случайно расположенной фигуры (квадрата внутри квадрата); <br>
&nbsp; &nbsp; **Данные:** &nbsp; При помощи numpy генерируем случайный набор данных в uint8. <br>
Используем pillow для сохранения полученных матриц в виде изображений. <br>
Прописываем собственный класс формирования датасета; <br>
&nbsp; &nbsp; **Модель:** &nbsp; Пишем класс полносвязной нейронной сети; <br>
&nbsp; &nbsp; **Функции активации:** &nbsp; ReLU; <br>
&nbsp; &nbsp; **Функция потерь:** &nbsp; MSELoss; <br>
&nbsp; &nbsp; **Конечный вид:** &nbsp; Пишем элементарную html-страничку. <br>
На fastapi прописываем backend часть и присоединяем frontend. <br>
Формируем docker-образ, разворачиваем контейнер, поднимаем локальный сервер.

<img width='800px' src='https://github.com/primera7790/pytorch_regression/blob/master/github_data/interface.PNG' alt='interface'/>

### Запуск:

1. Формирование docker-образа: &nbsp; `docker build . -t fast_pytorch_app:latest` из корневой папки;
2. Запуск docker-контейнера: &nbsp; `docker run -d -p 1234:8000 fast_pytorch_app` ;
3. Открываем страничку: &nbsp; `http://localhost:1234/`, либо `http://localhost:1234/docs` .


### Взаимодействие:

Ознакомление с текущими параметрами: &nbsp; кнопка `Get parameters`, либо `http://localhost:1234/parameters/get_params` ;
Изменение параметров: &nbsp; `http://localhost:1234/docs#/Parameters/set_params_parameters_set_params__parameter_name__post` , в поле `parameter_name` указываем расположение и имя параметра, отделяя уровни вложенности через `__`, в поле `new_value` указываем новое значение;
Запуск обучения модели: &nbsp; кнопка `Train`, либо `http://localhost:1234/operations/train` ;
Предсказание модели на тестовых данных: &nbsp; кнопка `Predict`, либо `http://localhost:1234/operations/predict` ;
Сгенерировать новые данные: &nbsp; кнопка `Create new data`, либо `http://localhost:1234/operations/data_creating` .

