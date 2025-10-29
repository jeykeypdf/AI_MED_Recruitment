# Recruitment Task – Classification of a Heart with Hypertrophic Cardiomyopathy (Cardiomegaly)

## Task Description

Your task is to prepare a solution for the **classification problem** of detecting **hypertrophic cardiomyopathy (cardiomegaly)** based on the provided features.  
You should use one of the **classical machine learning methods** described in the [ML.md](ML.md) file.  

The goal is to build a model capable of correctly distinguishing between a **healthy heart** and a **diseased heart** using the available data.

The repository contains a dataset stored in a **CSV** file, which includes selected geometric and imaging features used as the basis for further analysis.  
We expect the candidate to prepare code that:

1. Loads the data from the `task_data.csv` file.  
2. Splits the data into training and test sets.  
3. Performs preprocessing (e.g., standardization, normalization, etc.).  
4. Trains one or more selected models.  
5. Evaluates the solution using [cross-validation](https://www.geeksforgeeks.org/machine-learning/cross-validation-using-k-fold-with-scikit-learn/).  
6. Evaluates the solution on the test dataset (e.g., using accuracy, precision, recall, F1-score).  
7. Provides a brief description of the chosen approach.

---

## Data (CSV)

The first column contains a **photo ID**, and the second column indicates whether the heart was diagnosed with cardiomegaly:  
- `1` – positive diagnosis (diseased heart)  
- `0` – negative diagnosis (healthy heart)  

Below are the features describing the heart and lungs. Each feature includes its name (as used in the CSV file).

---

### Lung width

The horizontal distance between the outermost points of the lungs.

### Heart width

The maximum horizontal width of the heart.

### CR ratio

The ratio of heart width to lung width.

### Inertia tensors

Metrics describing the distribution of heart and lung pixels relative to the coordinate axes, capturing the shape and orientation of the objects.  
The `.csv` file contains four components of this feature:

* `xx` – distribution of pixels relative to the y-axis (elongation along x)  
* `yy` – distribution of pixels relative to the x-axis (elongation along y)  
* `xy` – distribution relative to both x and y axes (a high value indicates object rotation)  
* `normalized_diff` – a scalar value derived from the vector whose components are described above  

### Inscribed circle radius

The radius of the largest circle that can be inscribed within the heart area, describing its symmetry and compactness.

### Polygon area ratio

The ratio of the area of the polygon enclosing the heart contour to the actual heart area.

### Heart perimeter

The length of the heart contour.

### Heart area

The area occupied by the heart.

### Lung area

The area occupied by the lungs.

---

## Expectations

* The code should be written in **Python** (e.g., using libraries such as `scikit-learn`, `numpy`, `pandas`, `matplotlib`).  
* Include a short description of the solution in a **Markdown** (`.md`) file.  
* Present evaluation results clearly (e.g., results table, ROC/PR curves).  
* Code and commit messages should be written in **English**.  
  The Markdown description may be written in either **Polish** or **English**.  

To complete the task, please [Fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) this repository and submit the link to your fork **in your recruitment form**.

We recommend using **Jupyter Notebooks** for the implementation.

---

## Example model

In the [Example](Example/) folder, you will find a fully implemented machine learning project that demonstrates the entire workflow — from data loading and preprocessing to model training, evaluation, and final conclusions.
This example is designed to help beginner machine learning enthusiasts understand the structure of a complete project. It contains all the necessary information and practical guidance needed to successfully complete the recruitment task.

---

## Evaluation Criteria

* Correctness of the code.  
* Clarity and readability of the implementation.  
* Use of classical machine learning methods.  
* Creativity of the solution.  
* **Commit history in the repository** – clarity and consistency will also be evaluated.

---

**Good luck!**
