from lightgbm.sklearn import LGBMClassifier
clf = LGBMClassifier(
    learning_rate=0.05,
    n_estimators=10000,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=2019,
    device='gpu',
)
trn_x=[[1,2,3],[2,3,4]]*1000
trn_y=[0,1]*1000

clf.fit(
        trn_x, trn_y, 
        eval_set=[(trn_x, trn_y), (trn_x, trn_y)],
        early_stopping_rounds=100, verbose=5
    )
