import numpy as np
import pandas as pd
import gc

class AfeLib:
    
    def feature_engineering(self, data, suffix, groups, skip_columns):
        categorical_feats = []
        numeric_feats     = []
        to_dummy_feats    = []
        for f in data.columns:
            if f in skip_columns: continue
            if data[f].dtype == "object":
                categorical_feats.append(f)
                if len(data[f].value_counts()) <= 5:
                    to_dummy_feats.append(f)
            else:
                numeric_feats.append(f)

        for f in to_dummy_feats:
            data = pd.concat([data, pd.get_dummies(data[f], prefix=f+suffix)], axis=1)

        for f in categorical_feats:
            data[f], indexer = pd.factorize(data[f])

        mean_data = data.groupby(groups).mean()
        mean_data = mean_data.loc[:, numeric_feats]
        data = data.merge(right=mean_data.reset_index(), how="left", on=groups, suffixes=("", "_avg"+suffix))

        return data

if __name__ == '__main__':
    gc.enable()

    PATH = "/Users/pdesadmin/.kaggle/competitions/home-credit-default-risk"

    data = pd.read_csv(PATH+"/application_train.csv", nrows=100)
    test = pd.read_csv(PATH+"/application_test.csv", nrows=100)

    data["data_type"] = "train"
    test["data_type"] = "test"
    data = data.append(test)
    del test
    gc.collect()

    afe = AfeLib()
    data = afe.feature_engineering(data, "_application", ["SK_ID_CURR"], ["SK_ID_CURR", "TARGET", "data_type"])

    bureau_balance = pd.read_csv(PATH+"/bureau_balance.csv")
    bureau_balance = afe.feature_engineering(bureau_balance, "_bureau_balance", ["SK_ID_BUREAU"], ["SK_ID_BUREAU"])
    bureau = pd.read_csv(PATH+"/bureau.csv")
    bureau = afe.feature_engineering(bureau, "_bureau", ["SK_ID_CURR", "SK_ID_BUREAU"], ["SK_ID_CURR", "SK_ID_BUREAU"])
    bureau = bureau.merge(right=bureau_balance.reset_index(), how="left", on="SK_ID_BUREAU", suffixes=("", "bureau_balance"))
    del bureau_balance
    gc.collect()

    previous_application = pd.read_csv(PATH+"/previous_application.csv")
    previous_application = afe.feature_engineering(previous_application, "_previous_application", ["SK_ID_PREV", "SK_ID_CURR"], ["SK_ID_PREV", "SK_ID_CURR"])

    credit_card_balance = pd.read_csv(PATH+"/credit_card_balance.csv")
    credit_card_balance = afe.feature_engineering(credit_card_balance, "_credit_card_balance", ["SK_ID_PREV", "SK_ID_CURR"], ["SK_ID_PREV", "SK_ID_CURR"])
    previous_application = previous_application.merge(right=credit_card_balance.reset_index(), how="left", on="SK_ID_PREV", suffixes=("", "credit_card_balance"))
    del credit_card_balance
    gc.collect()

    installments_payments = pd.read_csv(PATH+"/installments_payments.csv")
    installments_payments = afe.feature_engineering(installments_payments, "_installments_payments", ["SK_ID_PREV", "SK_ID_CURR"], ["SK_ID_PREV", "SK_ID_CURR"])
    previous_application = previous_application.merge(right=installments_payments.reset_index(), how="left", on="SK_ID_PREV", suffixes=("", "installments_payments"))
    del installments_payments
    gc.collect()

    POS_CASH_balance = pd.read_csv(PATH+"/POS_CASH_balance.csv")
    POS_CASH_balance = afe.feature_engineering(POS_CASH_balance, "_POS_CASH_balance", ["SK_ID_PREV", "SK_ID_CURR"], ["SK_ID_PREV", "SK_ID_CURR"])
    previous_application = previous_application.merge(right=POS_CASH_balance.reset_index(), how="left", on="SK_ID_PREV", suffixes=("", "POS_CASH_balance"))
    del POS_CASH_balance
    gc.collect()

    data = data.merge(right=previous_application.reset_index(), how="left", on="SK_ID_CURR", suffixes=("", "previous_application"))
    del previous_application
    gc.collect()

    data[data["data_type"]=="train"].to_csv("./application_train_result.csv", encoding="utf-8", index=False)
    data[data["data_type"]=="test"].to_csv("./application_test_result.csv", encoding="utf-8", index=False)
