# 🚢 TitanicClassifierNN – Predict who survived!

This project was created as part of the **Neural Networks Basics** course in the *Automation Systems Engineering* program. Our goal was to build a model that predicts which Titanic passengers survived, using a **deep neural network with custom callbacks and thorough metric analysis**.

---

## 🎯 Project Goal

The task was to solve a classic binary classification problem based on the real Titanic dataset from Kaggle: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/overview).

> Can we predict who had a chance to survive the Titanic disaster? We answer this question using deep neural networks!


---

## 🔬 Data Approaches

We tested 3 different approaches to data preprocessing:

| Approach | Characteristics |
|----------|-----------------|
| **1** | Removed `Cabin`, analyzed `Ticket` prefixes, added ticket group size |
| **2** | Removed `Cabin` and `Ticket`, no prefix encoding – **most optimal** |
| **3** | Added `IsAlone` and `FamilySize` features, simplified structure |

**Approach 2** delivered the best results, being the most stable and generalizable.

---

## 🧠 Neural Network Models

We designed multiple versions of a Multi-Layer Perceptron (MLP), differing in depth, dropout, and loss functions:

- `model1` – classic setup with `binary_crossentropy`
- `model2a` – smaller model with regularization and `early stopping`
- `model2b` – model with `focal_loss`, better for imbalanced data
- `model3` – medium model with added features (FamilySize/IsAlone)

---

## ⚙️ Training & Metrics

During training, we used:
- Custom callbacks for F1 Score, Precision, and ROC-AUC
- `EarlyStopping`, `Dropout`, `StandardScaler`
- TensorBoard for real-time metric tracking

### Validation metrics:

| Model    | Accuracy | F1 Score | Precision | ROC-AUC | Notes                             |
|----------|----------|----------|-----------|---------|-----------------------------------|
| Model 1  | **0.85** | **0.75** | 0.79      | 0.88    | Highest precision and accuracy    |
| Model 2a | 0.83     | 0.74     | 0.76      | 0.87    | Fast convergence                  |
| Model 2b | 0.84     | 0.74     | 0.73      | **0.89**| Best AUC, robust on minority class|

---

## 📈 Analysis & Visualizations

All models were thoroughly analyzed. We generated:

- 📉 Loss and accuracy plots  
- 📊 F1 Score, ROC-AUC, and Precision graphs  
- 🧮 Confusion matrices and classification errors  
- 🧠 Neural network weight visualizations  
- ❌ Misclassified sample reports  

---

## 📌 Conclusions

- **Model 1** achieves the highest precision and performs best with the majority class  
- **Model 2b**, thanks to `focal_loss`, is better at detecting the minority `Survived` class  
- Both models achieved **great performance without overfitting**, thanks to early stopping and regularization  
- The project shows that thoughtful preprocessing has a **major impact on model stability and accuracy**

> ✅ A complete and scalable classification pipeline built with Keras and custom metrics.

---

## 👥 Project Information

This project was developed as a **group assignment** by a team of three students as part of the course **Neural Networks Basics**.

- 🎓 Field of Study: *Automation Systems Engineering*  
- 🗓️ Academic Year: 2024/2025  
- 🏫 Course Type: Practical group project  
- 👨‍👩‍👧 Team: 3 members (Wiktoria Michalska, Grzegorz Cyba, Patryk Kurt)

---

## 🛠️ How to Run the Project

```bash
pip install -r requirements.txt
python main.py

