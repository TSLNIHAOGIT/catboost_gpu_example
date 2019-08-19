from catboost import CatBoostClassifier

train_data = [[0, 3],
              [4, 1],
              [8, 1],
              [9, 1]]*1000
train_labels = [0, 0, 1, 1]*1000

model = CatBoostClassifier(
                           iterations=3000,learning_rate=0.05,max_depth=11,l2_leaf_reg=1,verbose=10,early_stopping_rounds=400,task_type='GPU',eval_metric='F1',
                           devices='0:1',
                           #iterations=1000, 
                           #task_type="GPU",
                           #verbose=10
                          )
model.fit(train_data,
          train_labels,
          verbose=10)
