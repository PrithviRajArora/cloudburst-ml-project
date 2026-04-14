# CloudBurst Prediction using Machine Learning

A machine learning project that builds a binary classification system to predict whether a cloudburst will occur the next day, using today's meteorological readings as input features.

---

## Project Structure

```
Project 3/
|-- CloudBurst_ML_Project.ipynb      # Main Jupyter Notebook (all 4 phases)
|-- cloudpredictionsystemproject.csv # Dataset (145,460 rows x 23 columns)
|-- README.md                        # This file
|-- Report.md                        # Full project report
|-- requirements.txt                 # Python dependencies
```

---

## Dataset

- Source: Kaggle CloudBurst Dataset: https://www.kaggle.com/datasets/akshat234/cloudburst
- - Size: 145,460 rows, 23 columns
  - - Coverage: Multiple Australian cities (Albury, etc.)
    - - Target Variable: CloudBurstTomorrow
     
      - The dataset file cloudpredictionsystemproject.csv is included in this repository.
     
      - ---

      ## Requirements

      Python version: 3.8 or above

      Install all dependencies using:
      ```bash
      pip install -r requirements.txt
      ```

      ---

      ## How to Run

      Option 1 - Jupyter Notebook (Recommended)

      - Clone the repository: git clone https://github.com/PrithviRajArora/cloudburst-ml-project.git
      - - Install dependencies: pip install -r requirements.txt
        - - Launch Jupyter: jupyter notebook
          - - Open CloudBurst_ML_Project.ipynb
            - - Go to Kernel > Restart and Run All
             
              - Option 2 - VS Code with Jupyter Extension
             
              - - Open the project folder in VS Code
                - - Install the Jupyter extension
                  - - Open CloudBurst_ML_Project.ipynb
                    - - Click Run All
                     
                      - ---

                      ## Project Phases

                      The notebook is organised into four phases corresponding to the project rubric:

                      | Phase | Description |
                      |-------|-------------|
                      | Phase 1 | Data Understanding and Preprocessing |
                      | Phase 2 | Supervised Learning |
                      | Phase 3 | Optimisation and Unsupervised Learning |
                      | Phase 4 | Advanced Models |

                      ---

                      ## Results Summary

                      | Model | Accuracy | F1 Score | AUC-ROC |
                      |-------|----------|----------|---------|
                      | Logistic Regression (Optimised) | ~0.77 | ~0.61 | ~0.82 |
                      | Random Forest (Tuned) | ~0.82 | ~0.67 | ~0.88 |

                      Final model selected: Tuned Random Forest.

                      ---

                      ## Report

                      See Report.md for the full written project report.

                      ---

                      ## Course Information

                      - Course: Machine Learning
                      - - Institution: Jagran Lakecity University (JLU)
                        - - Semester: 6
                          - - Student: Prithviraj Arora (2023BTCSE025L)
                            - 
