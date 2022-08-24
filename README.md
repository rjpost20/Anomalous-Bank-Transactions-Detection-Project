# Phase 5 Project: *Detecting Anomalous Financial Transactions*

<img src="https://github.com/rjpost20/Anomalous-Bank-Transactions-Detection-Project/blob/main/data/AdobeStock_319163865.jpeg?raw=true">
Image by <a href="https://stock.adobe.com/contributor/200768506/andsus?load_type=author&prev_url=detail" >AndSus</a> on Adobe Stock

## By Ryan Posternak

### Links

Presentation slidedeck PDF:


*This is an assignment for learning purposes. FinCEN is not involved with this project in any way.*

<br>

## Overview and Business Understanding

Money laundering is a <a href="https://www.fincen.gov/what-money-laundering" > major global concern</a> for national governments and financial institutions. It represents a threat not only for the agencies charged with policing such activity, but also the industry participants themselves, who face reputational damage, fines and increased regulatory scrutiny should it be revealed that bad actors used their networks for illicit financial activity. The UN Office on Drugs and Crime <a href="https://www.unodc.org/unodc/en/money-laundering/overview.html" > estimates </a> that money laundering encompasses 2 - 5% of global GDP, representing \$800 billion - \$2 trillion in 2022 USD. The specific motivations for money laundering are numerous, but can include terrorist financing, proliferation financing, and attempted concealment of funds obtained from theft and major crimes.

One promising tool in the fight against money laundering is the use of machine learning models to detect anomalous financial transactions. When such models flag a transaction, further investigation can be conducted to determine if the activity indeed represents illegal criminal activity or not. The accuracy of such models, however, is paramount, as the limited resources of banks and regulatory agencies could not possibly hope to investigate more than a handful of the <a href="https://www.federalreserve.gov/paymentsystems/fedach_yearlycomm.htm" > tens of millions</a> of transactions that occur daily.

![image](https://user-images.githubusercontent.com/105675055/186501449-7392de52-dd31-4bcf-81ce-8f85b71e5bcc.png)
Image by <a href="https://www.unodc.org/unodc/en/money-laundering/overview.html" >UN Office on Drugs and Crime </a>

The goal and motivation of this project is to provide the <a href="https://www.fincen.gov/" > Financial Crimes Enforcement Network</a>, aka FinCEN (*project is for academic purposes - FinCEN not actually involved*) with a machine learning model that can process routine financial transaction data and classify whether the transaction is anomalous or not. Such a model can be used not only by FinCEN and other regulatory agencies, but also by industry participants (i.e. financial institutions) to proactively detect and deter illicit activity being conducted through their systems.

