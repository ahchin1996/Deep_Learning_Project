# Deep_Learning_Project


ANN
------------
本研究針對人口普查資料進行年收入是否大於五萬 美金 以及每周工作時數之預測，透過演算法訓練資料進行測試資料的驗證準確度與降低誤差。隨著大量資料快速增長與累積以及機器學習與深度學習蓬勃發展、大數據分析已經成為學術界及各行各業中的關注與學習。從海量的資料中篩選出有用的資訊，並進行資料的預測即是機器學習領域方面積極所做的研究。本研究將使用 Python撰寫類神經網路(Artificial Neural Network)進行程式訓練模型在演算法的訓練之下在年收入是否大於五萬美元的預測上以準確度作為預測績效達到 0.85以及損失函數 0.32在每周工作時數預測以均方根誤差績效評估達到 10.82。


Aiwen Mango Rank Classification Competition
------------
愛文芒果採收後依品質篩選為 A、B、C三等級，依序為出口用、內銷用、加工用。然而愛文芒果現階段仍依靠人工篩選，這現象除了農村人口流失導致人力短缺，篩選芒果流程也因保鮮期壓縮地極短，導致篩選芒果階段約有10%的誤差，若以外銷金額估計，每年恐怕損失1600萬台幣，若能改善此部分則能大幅降低銷售成本及提高獲利及市占率。本研究 透過演算法訓練資料進行測試資料的驗證準確度與降低誤差 。 使用 Python撰寫類神經網路 (Artificial Neural Network)進行 程式訓練模型採用BIIC Lab所提供之5600筆芒果之圖像。透過此資料集來預測芒果的等級，使用 DenseNet的預訓練模型訓練之下芒果的分類預測，並在 1600張測試資料集上績效達到 0.7575。

Aiwen mango is selected as grade A, B and C according to its quality after being harvested, which are applied for export, domestic sale and processing. However, Aiwen mango still relies on manual screening at the present stage. In addition to the shortage of manpower caused by the loss of rural population, its selection process also has an error of about 10% due to the extremely short compressed shelf life. If we estimate the amount of export sales, we supposedly lose NT $16 million per year. If we can improve this part, we can significantly reduce the cost of sales and increase the profit and market share. In this study, the algorithm was used to train the data to verify the accuracy and reduce the error of test data. Artificial Neural Network was used to train the model, and 5600 mango images provided by BIIC Lab were used. This data set was applied to predict the grade of mango. The classification and prediction of mango were trained by DenseNet's pre-training model, then the performance reached 0.7575 (accuracy) on 1600 test data sets.

CNN_VGG
------------
本研究利用CIFAR10的資料集與VGG Face2資料集，分別進行圖片分類與人臉辨別，藉由卷曲類神經網路演算法進行訓練，本研究使用了3種不同的卷曲類神經架構，當中包含 Vgg16預訓練模型與simple Resnet模型。在影像辨識的熱潮下，許多關於影像辨識的架構與論文相繼出現，像是 GoogleNet、VGG19、ResNet等，為此本研究選取普遍績效較好的架構當作訓練模型，建立出 10種類行的辨識模型與20個不同人的人臉辨識，績效分別為 0.8974以及 0.90。
