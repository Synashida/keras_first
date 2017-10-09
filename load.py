from keras.models import load_model
import evaluate_data
import numpy as np

model = load_model('model.h5')

eX = evaluate_data.eX
eY = evaluate_data.eY

#学習結果の確認
ret = model.predict_classes(eX)

collectCount = 0
# ラベル毎の正解の保存
collects = [[0, 0],[0, 0],[0, 0], [0, 0]]
for idx, val in enumerate(ret):
  print("予測:{0},実際{1},{2}".format(val, eY[idx], val == eY[idx]))
  collectCount = collectCount + (1 if val == eY[idx] else 0)
  collects[eY[idx]][0] = collects[eY[idx]][0] + 1
  if val == eY[idx]:
    collects[eY[idx]][1] = collects[eY[idx]][1] + 1
    


print("正解率{0}% (検証日数 {1}日)".format(collectCount/len(ret)*100, len(ret)))
print("雨の正解率 = {0}%\n晴れの正解率 = {1}%\n曇の正解率 = {2}%\n雪の正解率 = {3}%".format(
  collects[0][1] / collects[0][0] * 100 if collects[0][0] > 0 else 0,
  collects[1][1] / collects[1][0] * 100 if collects[1][0] > 0 else 0,
  collects[2][1] / collects[2][0] * 100 if collects[2][0] > 0 else 0,
  collects[3][1] / collects[3][0] * 100 if collects[3][0] > 0 else 0
))
