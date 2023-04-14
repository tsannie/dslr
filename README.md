# dslr 🧙

![image](https://i.imgur.com/l2tZKUo.jpg)

## 📝 Description

dslr (*data science logistic regression*) is a project that aims to predict the house of Hogwarts a student will be in based on student's results in different courses. The goal is to use [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) to predict the house of a student based on the results of the student in different courses.

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

This script trains the [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) model. It saves the model in a pickle file.
For the training, I used the gradient descent algorithm.



For my cost function ([loss function](https://en.wikipedia.org/wiki/Loss_function)) I used the following formula:

$ J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) $








