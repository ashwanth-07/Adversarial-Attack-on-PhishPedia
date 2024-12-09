# Adversarial Attacks on PhishPedia  

This repository implements adversarial attack techniques, including **PGD (Projected Gradient Descent)** and **FGSM (Fast Gradient Sign Method)**, on **PhishPedia** — a state-of-the-art, two-stage visual phishing detection framework.

PhishPedia combines:  
1. **Logo Detection:** Identifies logos within incoming phishing webpages.  
2. **Siamese Network:** Compares the detected logos with a trusted logo database to determine the legitimacy of the webpage.  

For more details on PhishPedia, visit the original [PhishPedia repository](https://github.com/lindsey98/Phishpedia).  

---

## Features  

- **Adversarial Attack Implementation:** Evaluate the robustness of PhishPedia against popular adversarial attack methods:  
  - **FGSM:** Introduces small perturbations to input images based on the gradient of the loss function.  
  - **PGD:** Performs iterative, fine-grained perturbations to maximize the impact on model predictions.  

- **Attack Evaluation:** Measure how attacks affect logo detection accuracy and the Siamese model’s matching performance.  

---

## Dataset for Testing  

Use the following dataset for testing adversarial attacks on PhishPedia:  
[Download the dataset here](https://drive.google.com/file/d/12ypEMPRQ43zGRqHGut0Esq2z5en0DH4g/view)  

Ensure you download and extract the dataset into the appropriate directory before running the scripts.  

---

## References  

- **PhishPedia:** [GitHub Repository](https://github.com/lindsey98/Phishpedia)  
- **Adversarial Attacks:**  
  - Ian J. Goodfellow et al. (2015), *Explaining and Harnessing Adversarial Examples*  
  - Aleksander Madry et al. (2018), *Towards Deep Learning Models Resistant to Adversarial Attacks*   
