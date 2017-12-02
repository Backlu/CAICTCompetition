# CAICTCompetition

- 資料欄位篩選：透過上網研究風機相關的Paper, 篩選和風機結冰相關的欄位：
- 功率/風速/對風角/偏航位置/環境溫度/平均風向角
- 訓練資料擴增：先從測試集做第一次簡單的預測, 把預測結果放回訓練集訓練model.
- DL 模型Model: fully connect NN (3 hidden layer)
- 訓練20個神經元個數隨機的NN, 將預測結果取平均
