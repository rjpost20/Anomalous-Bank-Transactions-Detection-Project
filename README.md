# Phase 5 Project: *Detecting Anomalous Financial Transactions*

<img src="https://github.com/rjpost20/Anomalous-Bank-Transactions-Detection-Project/blob/main/data/AdobeStock_319163865.jpeg?raw=true">
Image by <a href="https://stock.adobe.com/contributor/200768506/andsus?load_type=author&prev_url=detail" >AndSus</a> on Adobe Stock

## By Ryan Posternak

### Links

<a href="https://github.com/rjpost20/Anomalous-Bank-Transactions-Detection-Project/blob/main/presentation_slide_deck.pdf" >Presentation slidedeck PDF </a>


*This is an assignment for learning purposes. FinCEN is not involved with this project in any way.*

<br>

## Overview and Business Understanding

Money laundering is a <a href="https://www.fincen.gov/what-money-laundering" > major global concern</a> for national governments and financial institutions. It represents a threat not only for the agencies charged with policing such activity, but also the industry participants themselves, who face reputational damage, fines and increased regulatory scrutiny should it be revealed that bad actors used their networks for illicit financial activity. The UN Office on Drugs and Crime <a href="https://www.unodc.org/unodc/en/money-laundering/overview.html" > estimates </a> that money laundering encompasses 2 - 5% of global GDP, representing \$800 billion - \$2 trillion in 2022 USD. The specific motivations for money laundering are numerous, but can include terrorist financing, proliferation financing, and attempted concealment of funds obtained from theft and major crimes.

![image](https://user-images.githubusercontent.com/105675055/186501449-7392de52-dd31-4bcf-81ce-8f85b71e5bcc.png)
Image by <a href="https://www.unodc.org/unodc/en/money-laundering/overview.html" >UN Office on Drugs and Crime </a>

One promising tool in the fight against money laundering is the use of machine learning models to detect anomalous financial transactions. When such models flag a transaction, further investigation can be conducted to determine if the activity indeed represents illegal criminal activity or not. The accuracy of such models, however, is paramount, as the limited resources of banks and regulatory agencies could not possibly hope to investigate more than a handful of the <a href="https://www.federalreserve.gov/paymentsystems/fedach_yearlycomm.htm" > tens of millions</a> of transactions that occur daily.

The goal and motivation of this project is to provide the <a href="https://www.fincen.gov/" > Financial Crimes Enforcement Network</a>, aka FinCEN (*project is for academic purposes - FinCEN not actually involved*) with a machine learning model that can process routine financial transaction data and classify whether the transaction is anomalous or not. Such a model can be used not only by FinCEN and other regulatory agencies, but also by industry participants (i.e. financial institutions) to proactively detect and deter illicit activity being conducted through their systems.

<br>

## Data Sources

Data for this project was obtained from <a href="https://www.drivendata.org/" >drivendata.org </a> as part of their PETs prize challenge: financial crime track. PETs, or privacy-enhancing technologies, involves finding new and innovative ways to harness the power of data while adhering to proper respect and safeguards for privacy, security and confidentiality of the data and its sources.

The data itself was produced through a collaboration between the National Institute of Standards and Technology (NIST) and the National Science Foundation (NSF). Due to the security and confidentiality concerns of such a sensitive matter, the dataset is synthetic, though it is intended to represent global payments data in broad strokes. The transactions dataset contains a binary discrete numerical `Label` column, the target variable for the project, with a `0` representing a non-anomalous transaction and a `1` representing an anomalous transaction.

![Slide 4](https://user-images.githubusercontent.com/105675055/186509494-7978b387-7a4c-4af5-a7e9-1763911bbb26.jpg)

<br>

## Exploratory Data Analysis and Feature Engineering

The transactions dataset consisted of 15 `string` features, two `integer` features, two `double`/`float` features, and one `timestamp` feature. The bank accounts dataset consisted of five `string` features and one `integer` feature. Because of the prevalence of `string` features in both datasets, a good deal of preprocessing and feature engineering was required in order to extract as much signal from the data as possible.

As part of the exploratory data analysis phase, differences in feature prevalence between non-anomalous and anomalous transactions were examined. Some trends that emerged among multiple predictor variables were either a smaller grouping of possible values among anomalous transactions as compared to non-anomalous ones (exemplified below), or the opposite: a larger grouping of possible values among anomalous transactions with more rare values not being commonly found among non-anomalous transactions.

![sender_banks_barplot](https://user-images.githubusercontent.com/105675055/186523034-ab994c1e-3f0b-470d-bdcb-5420d262d7a9.png)

The following features were engineered in order to maximize the signal captured in the two datasets:

- **`OrderingCountry` and `BeneficiaryCountry`**: These features were created by extracting the two-letter country code from the `OrderingCountryCityZip` and `BeneficiaryCountryCityZip` columns of the transactions dataframe. These two features were used for the `OrderingCountryFreq` and `BeneficiaryCountryFreq` numerical features below, as well as categorical features themselves in the full dataset via one hot encoding.

- **`InstructedAmountUSD`**: There were eight unique currencies used among the transactions in the dataset, and the `InstructedAmount` and `SettlementAmount` features in the original dataset were in the scale of the currencies used in the individual transactions. In order to get all transactions on a consistent scale, this new feature was engineered by converting each transaction currency amount to its USD equivalent on 2022/01/12, the median transaction date in the 23-day window of the transactions dataset.

![instructed_amount_usd_hist](https://user-images.githubusercontent.com/105675055/186510886-291ce82d-dbf2-4a04-97f5-35ec6e5ffeff.png)

- **`IntermediaryTransactions`**: While each row/observation in the dataset represented a unique individual transaction, some individual transactions represented one part of a end-to-end transaction, which could be identified by their UETR codes. This feature was created by grouping and counting UETR code occurrences, and subtracting 1 to obtain the number of intermediary transactions for each end-to-end transaction.

- **`OriginalSender` and `FinalReceiver`**: The `Sender` and `Receiver` bank features in the original dataset represented the sender and receiver banks of the individual transactions, but for end-to-end transactions where intermediary banks were used, the `Sender` and `Receiver` bank may not have represented the original sender bank and final receiver bank in the end-to-end transaction (which was the correct way to associate them according to the data providers). This feature was created to address this discrepancy.

![sender_bank_type_barplot](https://user-images.githubusercontent.com/105675055/186516557-482cf398-ae3d-41f7-a1c8-12bd21b40040.png)

- **`Flagged`**: The main feature of value in the bank accounts dataframe was the `Flags` feature, which contained one of 11 unique potential flags associated with each account in the transactions dataframe (save for a very small subset of accounts with no associated flag). The vast majority of accounts had a flag of `0`, which signified "No flags", while the other flags may have represented an account being closed, a name mismatch, or an account under monitoring, to name a few. This feature of the bank accounts dataframe was joined with the main transactions dataset using a SQL join operation in PySpark. Additionally, as EDA revealed that any account with a flag other than `0` meant that the associated transactions were anomalous, this feature was converted into binary discrete numerical form: `0` if the associated flag was `0` and `1` if the associated flag was any flag other than `0`.

- **`OrderingCountryFreq` and `BeneficiaryCountryFreq`**: Exploratory data analysis of the training data revealed that anomalous transactions tended to be concentrated among a smaller subset of more frequently appearing countries. In order to capture this signal (especially among the numeric dataset where categorical country features were not used) these features were created. 

![beneficiary_countries_barplot](https://user-images.githubusercontent.com/105675055/186516167-b8a9dd1c-3573-48f9-9e2e-de30eb32be01.png)

- **`Hour` and `SenderHourFreq`**: The `Hour` feature was created by extracting the transaction hour from the `Timestamp` feature of the transactions dataset, and the `SenderHourFreq` feature was created by calculating the frequency of each hour among all transactions.

- **`SenderCurrencyFreq` and `SenderCurrencyAmtAvg`**: The `SenderCurrencyFreq` feature was created by matching the `OriginalSender` bank with the currency used in each transaction, while the `SenderCurrencyAmtAvg` feature was created by matching the `OriginalSender` feature with the `InstructedAmountUSD` feature. Sender banks were used rather than receiver banks as EDA revealed greater differences in sender bank between non-anomalous and anomalous transactions than receiver bank.

- **`SenderFreq` and `ReceiverFreq`**: These features were created by calculating the frequency of the `OriginalSender` and `FinalReceiver` associated with each individual transaction.

- **`SenderReceiverFreq`**: This feature was created by finding all unique combinations of `SenderBank` and `ReceiverBank` in every transaction of the dataset and calculating the frequency of each.

<br>

## Modeling and Results



![Slide 8](https://user-images.githubusercontent.com/105675055/186525521-8c534dfb-62ca-4cd9-95f8-8ed351ab38e8.jpg)


<br>

## Conclusions, Recommendations, and Next Steps

***Conclusions:***

- **Final model strengths and weaknesses:** Our highest scoring model, model 7.4 - multilayer perceptron (artifical neural network), achieved a test F1 score of 0.567, beating the previous best model, a decision tree, by 0.017. The model correctly classified approximately 41% of the anomalous transactions (recall) and had a positive predictive value (precision) of over 93%, meaning over 93% of its predictions of anomalous transactions are in fact anomalous. While correctly identifying less than half of the anomalous transactions may seem mediocre, given that the model essentially needs to find needles in a haystack (761 anomalous transactions out of over 705k) with near surgical precision, it is no small feat.<br>

- **After a certain point, it is difficult to increase recall without substantially reducing precision.** As we outlined in the business understanding section, because this dataset is so imbalanced, even a 1% loss in precision on the testing data means ~700 additional false positives, nearly the entire amount of true positives. This means that increasing recall tends to come at a heavy cost in terms of reduced precision and more false positives, which ultimately seriously degrades the value of the model.<br>

- **The most important features for the model include whether the account is flagged, the country of the beneficiary account, and the frequency of the country of the ordering entity.** Sender currency frequency and beneficiary country frequency round out the top five. It's not surprising that the `Flagged` feature was important, given that we saw it was perfectly correlated with anomalous transactions in the EDA section. The country of the beneficiary account is also a logical feature to contain a good deal of predictive value. Perhaps surprisingly, `SenderReceiverFreq` was a very poor predictor, at least for the random forest model. `SenderFreq` and `SettlementCurrency` also did not seem to contain much predictive value.<br>

***Recommendations and Use Cases:***

- **Use the models as a guide on where to focus investigatory resources.** The limited time and resources of investigators should be wisely spent so as to limit time chasing false leads or annoying law-abiding customers. A model which can identify truly anomalous/potentially illicit transactions has a lot of value in terms of providing a signpost for where stakeholders should focus their resources for maximum benefit.<br>

- **Improve precision scores before using one of the models with a higher recall but lower F1 score.** As we saw, some models had higher recall than the best perceptron or decision tree models, but came with much lower precision. These models may have potential given additional tuning, but as is they likely carry more downside than upside in terms of false positives and have the potantial to lead to time chasing dead ends.<br>

- **Non-linear models appear to outperform linear ones for this use case.** Many of our initial linear models suffered from overreliance on one or a few features, likely because they were the only ones with a clear linear correlation with the target variable. As we saw in the EDA section though, many of the features do not have a linear relationship with anomalous transactions. Better results will likely be achieved with models which can pick up on these more complex relationships, such as decision tree-based models and neural networks.<br>

***Next Steps and Remaining Questions:***

- **Obtain more positive class observations.** While synthetic adjustment to the dataset can be made to even out the class imbalance, such as over or undersampling, SMOTE, or adjusting class weights, the best option is almost always to obtain more observations of the minority class. The models could likely perform better if they have more data of the positive class to go off of.<br>

- **Experiment with synthetic minority oversampling technique (SMOTE).** SMOTE was not used as part of this project because it does not appear to be supported by PySpark at the time of writing. While we did oversample the minority class with some models, SMOTE may have done a better job preventing overfitting, as simple oversampling can tend to cause models to overfit to noise that happens to be present in the minority class observations.<br>

- **Why did the model fail to improve substantially after the inclusion of six additional categorical variables?** Despite the feature importances of the random forest model telling us that multiple one hot encoded features had high importance, the model barely budged in F1 score between the best numeric dataset model and the best full dataset model. Did the numeric features we engineered capture all or most of the signal in the categorical features? Or maybe some of the categorical features were indeed valuable, but others introduced noise into the model that drowned out the signal of the valuable predictors. Investigating this further could yield valuable insights.
