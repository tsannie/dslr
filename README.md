# dslr 🧙

![image](https://i.imgur.com/l2tZKUo.jpg)

## 📝 Description

dslr (_data science logistic regression_) is a project that aims to predict the house of Hogwarts a student will be in based on student's results in different courses. The goal is to use [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) to predict the house of a student based on the results of the student in different courses.

- [Data analysis](#-data-analysis)
  - [describe.py](#1---describepy)
  - [histogram.py](#2---histogrampy)
  - [scatter_plot.py](#3---scatter_plotpy)
  - [pair_plot.py](#4---pair_plotpy)
- [logreg_train.py](#🏋️-logreg_trainpy)
- [logreg_predict.py](#🤯-logreg_predictpy)

## 📦 Installation

```bash
git clone https://github.com/tsannie/dslr && cd dslr
pip install -r requirements.txt
```

## 🧐 Data analysis

I made some data visualization to understand the data better. You can find them in the `data_visualization` folder.

### 1 - describe.py

This script rebuilds the `describe` function of pandas. It displays for each column the mean, standard deviation, minimum and maximum ...

![image](https://i.imgur.com/jhobmC1.png)

```bash
usage: describe.py [-h] csv_file

positional arguments:
  csv_file    file to describe (csv format)

optional arguments:
  -h, --help  show this help message and exit
```

### 2 - histogram.py

This script displays the histogram of the data. You can choose which courses to display by using the `-c` option.

![image](https://i.imgur.com/UPOsLWc.png)

```bash
usage: histogram.py [-h] [-c COURSES] csv_file

positional arguments:
  csv_file              Path to the csv file

optional arguments:
  -h, --help            show this help message and exit
  -c COURSES, --courses COURSES
                        List of courses to display separated by ','
```

### 3 - scatter_plot.py

This script displays the scatter plot of the data. You can choose which courses to display by using the `-c1` and `-c2` options.

![image](https://i.imgur.com/QddvVlL.png)

```bash
usage: scatter_plot.py [-h] [-c1 COURSE1] [-c2 COURSE2] csv_file

positional arguments:
  csv_file              Path to the csv file

optional arguments:
  -h, --help            show this help message and exit
  -c1 COURSE1, --course1 COURSE1
                        First course to compare
  -c2 COURSE2, --course2 COURSE2
                        Second course to compare
```

### 4 - pair_plot.py

This script displays the pair plot of the data. You can choose which courses to display by using the `-c` option.

![image](https://i.imgur.com/wxKpb45.png)

```bash
usage: pair_plot.py [-h] csv_file

positional arguments:
  csv_file    Path to the csv file

optional arguments:
  -h, --help  show this help message and exit
```

## 🏋️ logreg_train.py

This script trains the [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) model for each house. It saves the model in a csv file.
For the training, I used the gradient descent algorithm.
The training is threaded to be faster.

![image](https://i.imgur.com/TId7cRa.gif)

```bash
usage: logreg_train.py [-h] [-w weights_path] [-g] [-b batch_size]
                       [-l learning_rate] [-e epochs]
                       csv_file_path

positional arguments:
  csv_file_path         CSV file to train on

optional arguments:
  -h, --help            show this help message and exit
  -w weights_path, --weights weights_path
                        Path to save weights (default: ./data/thetas.csv)
  -g, --graph           Show graphs of training
  -b batch_size, --batch batch_size
                        Batch size (default: 10)
  -l learning_rate, --learning learning_rate
                        Learning rate (default: 0.01)
  -e epochs, --epochs epochs
                        Number of epochs (default: 12)
```

The option `-g` displays the graphs of the training. It shows the `loss` and the `accuracy` for each house.

Example for Gryffindor:
![image](https://i.imgur.com/ZJE6nWY.png)

## 🤯 logreg_predict.py

This script predict for each student in the csv file, the house he will be in based on the results of the student in different courses.
The most probable house is written in the `predictions.csv` file.
Example of predictions.csv:

```cs
0,Hufflepuff
1,Ravenclaw
2,Gryffindor
3,Hufflepuff
4,Hufflepuff
5,Slytherin
6,Ravenclaw
7,Hufflepuff
8,Ravenclaw
9,Hufflepuff
10,Hufflepuff
```

```bash
usage: logreg_predict.py [-h] [-s] [-p predictions_path]
                         dataset_path thetas_path

positional arguments:
  dataset_path         Dataset to predict
  thetas_path          Thetas to use for prediction

optional arguments:
  -h, --help           show this help message and exit
  -s, --show           Show predictions
  -p predictions_path  Predictions file path (default: ./data/predictions.csv)
```

when using the `-s` option, the script displays the predictions for each student:

![image](https://i.imgur.com/ImPqS1x.png)
