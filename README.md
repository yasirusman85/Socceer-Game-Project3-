# Football Match Outcome Prediction with EDA and Machine Learning

## Project Overview

This project aims to predict the outcome of football matches using various machine learning algorithms. It utilizes a dataset that contains historical match data, including match statistics like goals scored, possession, shots on target, etc., to train predictive models. We also perform exploratory data analysis (EDA) to understand the key factors influencing match outcomes and how different variables like possession, shots on target, or cards correlate with the likelihood of winning, losing, or drawing a match.

The following machine learning models are used in this project:
- Logistic Regression
- Random Forest Classifier
- XGBoost
- Support Vector Machine (SVM)

We evaluate the models based on metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

---

## Dataset

The dataset used in this project is from Kaggle:

[Football Match Dataset - Kaggle](https://www.kaggle.com/datasets/hugomathien/soccer)

It contains match-level data with features such as:
- Match outcome (Win, Loss, Draw)
- Home team and away team statistics (goals, possession, shots, etc.)
- Player performance metrics
- Additional match-related information

---

## Key Steps in the Project

### 1. Exploratory Data Analysis (EDA)

We performed several analyses to understand the key factors influencing match outcomes:

- **Key Factors Influencing Match Outcomes**:
    - We found that features such as **possession**, **shots on target**, and **red cards** correlate strongly with match outcomes.
    - Teams with higher possession and more shots on target tend to win matches, while red cards negatively impact the team's chances of winning.

- **Seasonal Trends**:
    - We explored whether there are seasonal trends in match outcomes, such as higher goal counts during certain months or tournaments. We also analyzed whether certain tournaments (league vs. cup) influence match outcomes.

- **Team and Player Performance**:
    - We identified high-performing teams and players across multiple seasons, especially those excelling in goal-scoring, assists, and defensive metrics.
    - Defensive metrics like **shots blocked** and **tackles** were also analyzed to compare team performance.

- **Anomalies in the Data**:
    - Outliers and anomalies such as unusually high or low goal counts were detected and analyzed. We also looked for unexpected outcomes related to factors like **red cards** or player injuries.

- **Home vs. Away Performance**:
    - We compared home team performance metrics like possession and goals scored against away teams and checked for statistically significant differences in match outcomes.

### 2. Machine Learning Model Selection

We split the dataset into training and testing sets (80/20 split) and trained several models to predict the match outcome:

- **Logistic Regression**
- **Random Forest Classifier**
- **XGBoost**
- **Support Vector Machine (SVM)**

### 3. Model Evaluation

Each model was evaluated based on the following metrics:
- **Accuracy**: Overall percentage of correct predictions.
- **Precision**: The ratio of true positives to the total predicted positives.
- **Recall**: The ratio of true positives to the total actual positives.
- **F1-Score**: The harmonic mean of precision and recall.
- **Confusion Matrix**: Visual representation of model predictions vs. actual outcomes.

Here are the results from the models:

- **Logistic Regression**:
    - Accuracy: 53.97%
    - Precision, Recall, F1-Score (for each class) were reported for Win, Loss, and Draw outcomes.

- **Random Forest Classifier**:
    - Accuracy: 48.72%
    - Precision, Recall, F1-Score for each class.

- **XGBoost**:
    - Accuracy: 54.87%
    - Precision, Recall, F1-Score for each class.

- **Support Vector Machine (SVM)**:
    - Accuracy: 53.87%
    - Precision, Recall, F1-Score for each class.

---

## Key Insights

1. **Match Outcome Prediction**: The model's accuracy varied between 48% and 54%, with different algorithms offering trade-offs between precision, recall, and overall performance.
2. **Key Factors Influencing Wins**: Features like **possession** and **shots on target** strongly correlate with the likelihood of winning a match. Red cards were also identified as a key factor negatively affecting match outcomes.
3. **Home vs. Away Performance**: Home teams generally had a slight advantage in terms of possession and goals scored, although the difference was not always statistically significant.
4. **Seasonal Trends**: Certain months and tournaments showed higher goal counts, but no definitive seasonal trend was observed in match outcomes.

---

## Setup Instructions

### Requirements

- Python 3.x
- Libraries:
    - `pandas`
    - `numpy`
    - `scikit-learn`
    - `xgboost`
    - `matplotlib`
    - `seaborn`
    - `requests`


## Conclusion

This project demonstrates the application of machine learning techniques to predict football match outcomes based on historical data. Through exploratory data analysis, we identified key factors that influence match outcomes, and several machine learning models were trained and evaluated to predict match results.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- The dataset is provided by Hugo Mathien on Kaggle. [Link to dataset](https://www.kaggle.com/datasets/hugomathien/soccer).
- Machine learning libraries used: `scikit-learn`, `xgboost`, `matplotlib`, and `seaborn`.

---

