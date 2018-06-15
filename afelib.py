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
            else:
                numeric_feats.append(f)

        for f in categorical_feats:
            data[f], indexer = pd.factorize(data[f])

        data, new_columns = self.one_hot_encoder(data)

        grouped   = data.groupby(groups)
        mean_data = grouped.mean()
        mean_data = mean_data.loc[:, numeric_feats]
        data = data.merge(right=mean_data.reset_index(), how="left", on=groups, suffixes=("", "_avg"+suffix))

        count_data = grouped.size()
        count_data = count_data.loc[:, categorical_feats]
        data = data.merge(right=count_data.reset_index(), how="left", on=groups, suffixes=("", "_count"+suffix))

        return data

    def target_encoder(self, data, target_variable):
        pass

    def one_hot_encoder(self, data, nan_as_category = True):
        original_columns = list(data.columns)
        categorical_columns = [col for col in data.columns if data[col].dtype == 'object' and len(data[col].value_counts()) <= 5]
        data = pd.get_dummies(data, columns= categorical_columns, dummy_na= nan_as_category)
        new_columns = [c for c in data.columns if c not in original_columns]

        return data, new_columns

    def kfold_lightgbm(df, num_folds, stratified = False):
        train_df = df[df['TARGET'].notnull()]
        test_df = df[df['TARGET'].isnull()]
        print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
        del df
        gc.collect()

        if stratified:
            folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
        else:
            folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)

        oof_preds = np.zeros(train_df.shape[0])
        sub_preds = np.zeros(test_df.shape[0])
        feature_importance_df = pd.DataFrame()
        feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV']]
        
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
            train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
            valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

            clf = LGBMClassifier(
                nthread=4,
                n_estimators=40000,
                learning_rate=0.01,
                num_leaves = 80,
                colsample_bytree=0.9497036,
                subsample=0.8715623,
                max_depth=40,
                reg_alpha=0.041545473,
                reg_lambda=0.0735294,
                min_split_gain=0.0222415,
                min_child_weight=39.3259775,
                silent=-1,
                verbose=-1,
            )

            clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
                eval_metric= 'auc', verbose= 100, early_stopping_rounds= 100)

            oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
            sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = feats
            fold_importance_df["importance"] = clf.feature_importances_
            fold_importance_df["fold"] = n_fold + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
            del clf, train_x, train_y, valid_x, valid_y
            gc.collect()

        print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)
        display_importances(feature_importance_df)
        return feature_importance_df


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

    bureau_balance = pd.read_csv(PATH+"/bureau_balance.csv", nrows=100)
    bureau_balance = afe.feature_engineering(bureau_balance, "_bureau_balance", ["SK_ID_BUREAU"], ["SK_ID_BUREAU"])
    bureau = pd.read_csv(PATH+"/bureau.csv", nrows=100)
    bureau = afe.feature_engineering(bureau, "_bureau", ["SK_ID_CURR", "SK_ID_BUREAU"], ["SK_ID_CURR", "SK_ID_BUREAU"])
    bureau = bureau.merge(right=bureau_balance.reset_index(), how="left", on="SK_ID_BUREAU", suffixes=("", "_bureau_balance"))
    del bureau_balance
    gc.collect()

    previous_application = pd.read_csv(PATH+"/previous_application.csv", nrows=100)
    previous_application = afe.feature_engineering(previous_application, "_previous_application", ["SK_ID_PREV", "SK_ID_CURR"], ["SK_ID_PREV", "SK_ID_CURR"])

    credit_card_balance = pd.read_csv(PATH+"/credit_card_balance.csv", nrows=100)
    credit_card_balance = afe.feature_engineering(credit_card_balance, "_credit_card_balance", ["SK_ID_PREV", "SK_ID_CURR"], ["SK_ID_PREV", "SK_ID_CURR"])
    previous_application = previous_application.merge(right=credit_card_balance.reset_index(), how="left", on="SK_ID_PREV", suffixes=("", "_credit_card_balance"))
    del credit_card_balance
    gc.collect()

    installments_payments = pd.read_csv(PATH+"/installments_payments.csv", nrows=100)
    installments_payments = afe.feature_engineering(installments_payments, "_installments_payments", ["SK_ID_PREV", "SK_ID_CURR"], ["SK_ID_PREV", "SK_ID_CURR"])
    previous_application = previous_application.merge(right=installments_payments.reset_index(), how="left", on="SK_ID_PREV", suffixes=("", "_installments_payments"))
    del installments_payments
    gc.collect()

    POS_CASH_balance = pd.read_csv(PATH+"/POS_CASH_balance.csv", nrows=100)
    POS_CASH_balance = afe.feature_engineering(POS_CASH_balance, "_POS_CASH_balance", ["SK_ID_PREV", "SK_ID_CURR"], ["SK_ID_PREV", "SK_ID_CURR"])
    previous_application = previous_application.merge(right=POS_CASH_balance.reset_index(), how="left", on="SK_ID_PREV", suffixes=("", "_POS_CASH_balance"))
    del POS_CASH_balance
    gc.collect()

    data = data.merge(right=previous_application.reset_index(), how="left", on="SK_ID_CURR", suffixes=("", "_previous_application"))
    del previous_application
    gc.collect()

    data[data["data_type"]=="train"].to_csv("./application_train_result.csv", encoding="utf-8", index=False)
    data[data["data_type"]=="test"].to_csv("./application_test_result.csv", encoding="utf-8", index=False)
