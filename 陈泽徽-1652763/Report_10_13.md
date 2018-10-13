# Report: Statistical Features with BiLSTM Experiments

Oct. 13

Zehui Chen

## Current Works

1. **Finished Most of Feature Engineering Works**
   - MA/ MA_Diff
   - EWM
   - 2nd.Exp.Smooth
   - LOWESS (unfinished, not know how to implement)
   - Diff
   - Wavelet (structure finished, but still have some problems with dimensions)
   - Holt-Winters/3rd.Exp.Smooth
   - AR/ARIMA

2. **Feature Engineering Data Feed into Model**

   1. New Feature

      1. Early stopping
      2. Learning rate reduce schedule
      3. Save the best model(checkpoint)
      4. Change the loss function from mse to binary_crossentropy

   2. Experiment Results

      The following results are based on dataset with KPI ID = 046ec29ddf80d62e. From the experiments below we can find that statistical features gives us an improvement of 6% on accuracy.

   | Results  | Original Data (None, 30, 1) | Generated Features Data (None, 30, 14) |
   | :------: | :-------------------------: | :------------------------------------: |
   | Accuracy |            93.1%            |                 99.8%                  |

   There is one thing to emphasize here: we utilized over sampling method provided by Zhu, which improved our positive ratio from 0.9% to 33.3%. Under this circumstance, it's ok for us to use accuracy to justify our model performance.

3. **Other Problems**

   - Oversampling method
   - LSTM need large computation

## Future Work

1. Create pipeline to transform all types of KPI data into a 14 dimensions(or even higher) vector that can be directly feed into our BiLSTM model.
2. Attempts to change the model and training process to get further progress.