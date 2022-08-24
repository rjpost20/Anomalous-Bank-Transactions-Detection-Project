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









