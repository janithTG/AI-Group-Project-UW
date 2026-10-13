# AI Group Project UW

## Manufacturing Efficiency Classification Using Machine Learning

MSc AI and Automation Group Project  
University West

## Project Overview

This project investigates machine learning methods for classifying manufacturing efficiency using operational, sensor, network, production, maintenance, and quality related data.

The prediction target is:

`Efficiency_Status`

The target contains three classes:

`Low`

`Medium`

`High`

The project follows a complete machine learning workflow including:

Dataset understanding

Exploratory data analysis

Data preprocessing

Baseline modelling

Model comparison

Cross validation

Feature importance analysis

Feature ablation

Final model selection

Evaluation on an untouched test dataset

The models investigated were:

`Majority Baseline`

`SGD Classifier`

`Decision Tree Classifier`

`Random Forest Classifier`

Because the target classes are strongly imbalanced, model evaluation does not rely on accuracy alone. Macro Precision, Macro Recall, Macro F1 and confusion matrices are also used throughout the project.

## Dataset

The dataset contains:

| Property | Value |
|---|---:|
| Observations | 100,000 |
| Original attributes | 13 |
| Unique machines | 50 |
| Missing values | 0 |
| Duplicate rows | 0 |

The target variable is:

`Efficiency_Status`

Its approximate class distribution is:

| Efficiency Status | Percentage |
|---|---:|
| Low | 77.8% |
| Medium | 19.2% |
| High | 3.0% |

This creates a significant class imbalance.

A classifier that always predicts `Low` can therefore achieve approximately 77.8% accuracy even though it completely fails to identify the Medium and High classes.

For this reason, Macro F1 is one of the main evaluation metrics used in the project.

## Input Features

The initial model uses manufacturing related measurements including:

`Temperature_C`

`Vibration_Hz`

`Power_Consumption_kW`

`Network_Latency_ms`

`Packet_Loss_%`

`Quality_Control_Defect_Rate_%`

`Production_Speed_units_per_hr`

`Predictive_Maintenance_Score`

`Error_Rate_%`

`Operation_Mode`

The following columns are not initially used as normal predictive features:

`Timestamp`

`Machine_ID`

`Machine_ID` is treated as an identifier rather than a continuous numerical measurement.

The raw `Timestamp` is also excluded from the final model after temporal patterns were investigated during exploratory analysis.

## Why the Project is Divided Into Six Notebooks

The project is intentionally divided into six notebooks instead of placing the complete workflow inside one large notebook.

Each notebook represents a distinct stage of the machine learning lifecycle.

This structure was selected for several reasons.

### Separation of Responsibilities

Dataset exploration, preprocessing, model training, evaluation, and final testing have different purposes.

Separating them makes each stage easier to understand and review.

### Reproducibility

Each notebook loads the required data and defines the variables required for that stage.

The notebooks therefore do not depend on hidden variables remaining in memory from another notebook.

### Easier Collaboration

The project is completed as a group project.

Separating the workflow allows group members to review, understand, and discuss individual stages without navigating one very large notebook.

### Easier Debugging

If a problem appears during preprocessing, model training, feature analysis, or final evaluation, it is easier to identify the responsible stage.

### Clear Experimental Progression

The notebooks follow the actual reasoning process of the project:

```text
Understand the data
        ↓
Explore relationships
        ↓
Prepare data and create baselines
        ↓
Compare machine learning models
        ↓
Investigate why the models perform well
        ↓
Evaluate the selected model once on unseen data
```

### Protection of the Final Test Set

The notebook structure also helps maintain experimental discipline.

The final test dataset is separated before model development and remains untouched during model comparison, feature importance analysis, and ablation experiments.

It is used only in Notebook 06 after the final model has already been selected.

## Notebook Workflow

### 01 Dataset Understanding

`notebooks/01_dataset_explore.ipynb`

This notebook focuses on understanding the raw dataset before machine learning is applied.

Main tasks include:

Dataset shape and column inspection

Data type inspection

Missing value analysis

Duplicate detection

Target class distribution

Class imbalance analysis

Machine ID analysis

Timestamp analysis

Descriptive statistics

Numerical correlation analysis

Class based feature comparisons

Initial investigation of Production Speed and Error Rate

An important finding from this stage was that the provided `Efficiency_Status` labels can be exactly reconstructed using threshold conditions involving Production Speed and Error Rate.

The observed relationship is:

```text
If Production Speed >= 400
and Error Rate <= 2
    High

Else if Production Speed >= 200
and Error Rate <= 5
    Medium

Else
    Low
```

This rule reproduced all 100,000 provided target labels.

This finding describes the structure of the dataset and should not be interpreted as proof that these variables cause manufacturing efficiency in real industrial environments.

### 02 Exploratory Data Analysis

`notebooks/02_exploratory_data_analysis.ipynb`

This notebook performs deeper exploratory analysis.

Main tasks include:

Numerical feature distributions

Skewness analysis

Box plots

Potential outlier analysis

Feature distributions across target classes

Production Speed analysis

Error Rate analysis

Production Speed and Error Rate scatter plots

Operation Mode analysis

Machine based analysis

Temporal exploration

Potential feature engineering

The exploratory analysis confirmed that Production Speed and Error Rate provide much stronger separation between the efficiency classes than the other available variables.

### 03 Preprocessing and Baseline Models

`notebooks/03_preprocessing_and_baseline_models.ipynb`

This notebook prepares the data for machine learning and establishes initial models.

The dataset is divided into:

`64%` training

`16%` validation

`20%` final test

Stratified splitting is used so that the Low, Medium, and High class proportions remain approximately consistent across each dataset.

The final test dataset is not used during model development.

Preprocessing includes:

`OneHotEncoder` for `Operation_Mode`

`StandardScaler` for numerical features used by the SGD Classifier

Scikit learn `ColumnTransformer`

Scikit learn `Pipeline`

Three initial models are evaluated:

`Majority Baseline`

`SGD Classifier`

`Decision Tree Classifier`

The Majority Baseline demonstrates why accuracy alone is misleading for the strongly imbalanced target.

### 04 Random Forest and Cross Validation

`notebooks/04_random_forest_and_cross_validation.ipynb`

This notebook introduces the Random Forest Classifier and evaluates model stability using three fold stratified cross validation.

The following models are compared:

| Model | CV Accuracy | CV Macro F1 |
|---|---:|---:|
| Majority Baseline | approximately 0.7783 | approximately 0.2918 |
| SGD Classifier | approximately 0.9077 | approximately 0.7954 |
| Decision Tree | approximately 0.99994 | approximately 0.99985 |
| Random Forest | approximately 0.99994 | approximately 0.99968 |

The Decision Tree achieved the strongest Macro F1 result and also showed very low variation across the cross validation folds.

Random Forest cross validation produced only four incorrect classifications among 64,000 observations.

The extremely strong performance of the tree based models is interpreted together with the threshold based target structure discovered during the earlier exploratory analysis.

### 05 Feature Importance and Ablation

`notebooks/05_feature_importance_and_ablation.ipynb`

This notebook investigates why the Decision Tree and Random Forest models achieve almost perfect performance.

Random Forest impurity based feature importance assigned approximately:

| Feature | Importance |
|---|---:|
| Error Rate | 66.47% |
| Production Speed | 31.71% |
| All remaining features combined | approximately 1.83% |

Production Speed and Error Rate together account for approximately:

`98.17%`

of the Random Forest impurity based feature importance.

Feature importance is not interpreted as causation.

A feature ablation experiment is then performed.

The Random Forest is evaluated first using all features and then after removing:

`Production_Speed_units_per_hr`

`Error_Rate_%`

Results:

| Experiment | Accuracy | Macro F1 |
|---|---:|---:|
| Random Forest with all features | approximately 0.9999 | approximately 0.9997 |
| Random Forest without Production Speed and Error Rate | approximately 0.7778 | approximately 0.2920 |

After the two dominant features are removed, Random Forest performance becomes approximately equivalent to the Majority Baseline.

The reduced Random Forest also fails to correctly classify any High efficiency observations.

A similar experiment using the Decision Tree produces the same general conclusion.

This provides strong evidence that the almost perfect model performance primarily depends on the target structure associated with Production Speed and Error Rate.

### 06 Final Model and Evaluation

`notebooks/06_final_model_and_evaluation.ipynb`

The final model is selected before the test dataset is accessed.

The Decision Tree is selected because it:

Achieved the highest cross validation Macro F1

Showed extremely stable cross validation performance

Was simpler than the Random Forest

Provided an interpretable representation of the threshold based target structure

After model selection, the training and validation portions are recombined.

The final model is therefore trained using:

`80,000 development observations`

It is then evaluated once on:

`20,000 previously unseen test observations`

Final Decision Tree characteristics:

| Property | Value |
|---|---:|
| Tree depth | 4 |
| Number of leaves | 5 |

Final test performance:

| Metric | Result |
|---|---:|
| Accuracy | 0.999850 |
| Macro Precision | 0.999870 |
| Macro Recall | 0.999268 |
| Macro F1 | 0.999569 |
| Weighted F1 | 0.999850 |

The final model correctly classified:

`19,997 / 20,000`

test observations.

Only three observations were incorrectly classified.

The remaining errors occurred very close to the previously identified target thresholds.

Two Medium observations had Error Rate values extremely close to `5`.

One High observation had Production Speed extremely close to `400`.

The final test performance is therefore highly consistent with the cross validation results.

No model parameters or features were changed after observing the final test results.

## Main Findings

The project produced several important findings.

### Class Imbalance Matters

The Low class represents approximately 77.8% of the dataset.

Therefore, accuracy alone can provide a misleading picture of model quality.

The Majority Baseline achieved approximately:

`77.8% accuracy`

but only:

`0.292 Macro F1`

This demonstrates the importance of class balanced evaluation metrics.

### Tree Based Models Strongly Outperform the Linear Model

The SGD Classifier substantially improves over the baseline.

However, Decision Tree and Random Forest models achieve almost perfect classification.

This is consistent with the threshold based nature of the target.

### Two Features Dominate the Prediction Problem

Production Speed and Error Rate provide almost all predictive information required to reconstruct `Efficiency_Status`.

The manually identified threshold rule reproduces all dataset labels.

Random Forest feature importance assigns approximately 98.17% combined importance to these two variables.

Removing them reduces Random Forest Macro F1 from approximately 0.9997 to 0.2920.

### High Accuracy Must Be Interpreted Carefully

The final accuracy of approximately 99.985% is extremely high.

However, the project does not interpret this number in isolation.

The analysis shows that the target variable contains a simple deterministic threshold structure.

Therefore, the high model performance does not necessarily mean that the classifier discovered complex industrial manufacturing behaviour.

Understanding why a model performs well is as important as reporting the performance itself.

## Evaluation Metrics

The main metrics used throughout the project are:

### Accuracy

The proportion of all predictions that are correct.

### Precision

Measures how often a predicted class is actually correct.

### Recall

Measures how many observations belonging to a class are successfully identified.

### F1 Score

Combines precision and recall into a single measure.

### Macro F1

Calculates F1 separately for Low, Medium, and High classes and gives each class equal importance.

Macro F1 is particularly important for this project because the target distribution is strongly imbalanced.

### Confusion Matrix

Shows which classes are predicted correctly and which classes are confused with each other.

## Repository Structure

```text
AI-Group-Project-UW/

data/
    raw/
        manufacturing_6G_dataset.csv
    processed/

notebooks/
    01_dataset_explore.ipynb
    02_exploratory_data_analysis.ipynb
    03_preprocessing_and_baseline_models.ipynb
    04_random_forest_and_cross_validation.ipynb
    05_feature_importance_and_ablation.ipynb
    06_final_model_and_evaluation.ipynb

figures/
    generated figures used during analysis

results/
    final_test_metrics.csv

models/
    final_decision_tree_pipeline.joblib

docs/
    project documentation

src/
    reusable source code if required

tests/
    project tests if required

requirements.txt

README.md
```

The project primarily uses Scikit learn pipelines rather than permanently generating multiple transformed dataset files.

This allows preprocessing to remain connected directly to the model and reduces the risk of applying inconsistent transformations.

## Setup

### Prerequisites

Python 3.10 or later is recommended.

Download Python from:

[python.org](https://www.python.org/downloads/)

On Windows, select **Add Python to PATH** during installation.

### 1. Clone the Repository

```bash
git clone https://github.com/janithTG/AI-Group-Project-UW.git
cd AI-Group-Project-UW
```

### 2. Create a Virtual Environment

#### macOS or Linux

```bash
python3 -m venv .venv
```

#### Windows

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

#### macOS or Linux

```bash
source .venv/bin/activate
```

#### Windows Command Prompt

```bat
.venv\Scripts\activate.bat
```

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

When the environment is active, the terminal normally begins with:

```text
(.venv)
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Register the Jupyter Kernel

This step is optional but recommended.

```bash
python -m ipykernel install --user --name=ai-group-project --display-name="Python (AI Group Project)"
```

Start Jupyter and select:

`Python (AI Group Project)`

as the notebook kernel.

For the complete workflow, begin with:

```text
notebooks/01_dataset_explore.ipynb
```

and execute the notebooks in numerical order.

### 6. Deactivate the Environment

```bash
deactivate
```

## Reproducing the Experiment

For a complete reproduction of the project, execute the notebooks in the following order:

```text
01
↓
02
↓
03
↓
04
↓
05
↓
06
```

Each notebook represents a logical stage of the experiment.

The raw dataset should remain unchanged inside:

```text
data/raw/
```

The final test dataset is generated using the same stratified split and fixed random seed used throughout the project.

`random_state = 42`

is used where applicable to improve reproducibility.

## Important Limitation

The most important limitation identified during this project is the structure of the supplied target variable.

`Efficiency_Status` can be reconstructed exactly using threshold conditions involving Production Speed and Error Rate.

Therefore, the near perfect Decision Tree and Random Forest performance should not automatically be interpreted as evidence that the models would achieve similar results on real manufacturing data.

The project conclusions describe the relationships found within this dataset.

Further validation using independently collected real industrial data would be required before making broader claims about real world manufacturing efficiency prediction.

## Generative AI Acknowledgement

Generative AI tools were used during the project to support explanation, code discussion, debugging, documentation drafting, and interpretation of machine learning concepts.

The group reviewed the generated suggestions, executed the experiments independently, inspected the resulting outputs, and based the reported conclusions on the actual experimental results.