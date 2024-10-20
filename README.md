# solid_stock_price_predictor
Everyone has to make a stock price predictor at one point this is my attempt slowly working to make it more competative!  

In this project, I developed an LSTM model to predict Apple's stock prices using historical data and technical indicators like the closing price, moving averages, and volume. My goal was to build a model that could provide accurate future stock price predictions. After making many tweaks to my model i was able to get to a solid RMSE of 3.18, which is performing well given that I only used a standard LSTM architecture and basic technical indicators. The final model effectively captured patterns in the data by focusing on recent trends while avoiding overfitting through a simplified two-layer LSTM structure.

The model's performance improved after I reduced the time step to 60 days, allowing it to focus on more recent price movements. I also simplified the LSTM architecture to avoid overfitting and switched to the Adam optimizer, which enhanced training and generalization. While I explored more complex strategies but for now I wanted to stick to the standard LSTM architecture.

Despite the solid performance, there’s potential for even further improvement. By using more advanced machine learning techniques, such as transformer models, attention mechanisms, or incorporating sentiment analysis from the media, I could potentially lower the RMSE to around 1.5-2.5, bringing the model closer to competing with state-of-the-art systems.
