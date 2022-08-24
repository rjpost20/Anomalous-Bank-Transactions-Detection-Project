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

![Slide 4](https://user-images.githubusercontent.com/105675055/186504784-3f2885ec-aa9b-44a8-ad3a-a9331fe63986.jpeg)



