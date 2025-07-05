# SentimentBayesLSTM: Hybrid Forecasting with GPT Insights and Bayesian Tuning

A sophisticated stock price prediction system that combines LSTM neural networks with sentiment analysis and Bayesian optimization for enhanced forecasting accuracy.

## 🎓 Academic Research

This project was developed as part of my dissertation research. The implementation demonstrates advanced machine learning techniques for financial forecasting, combining traditional technical analysis with modern AI approaches.

**📄 Full Research Paper**: If you're interested in the complete academic paper with detailed methodology, literature review, and comprehensive results analysis, please reach out to me directly.

**📋 Implementation Guide**: This repository includes a detailed PDF guide that walks through the entire implementation process, explaining each component and methodology used in the research.
## 🚀 Features

- **Advanced LSTM Architecture**: Multi-layer LSTM with bidirectional processing and attention mechanisms
- **Sentiment Analysis Integration**: Real-time news sentiment analysis using OpenAI's GPT models
- **Bayesian Hyperparameter Optimization**: Automated hyperparameter tuning using Gaussian Process optimization
- **Technical Indicators**: Comprehensive set of technical analysis indicators (RSI, MACD, Bollinger Bands, etc.)
- **Uncertainty Estimation**: Monte Carlo Dropout for prediction confidence intervals
- **Robust Data Processing**: Advanced data preprocessing with error handling and validation

## 📋 Requirements

### Python Dependencies

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib requests beautifulsoup4 openai tqdm yfinance scikit-optimize ta
```

### Required Libraries

- **Data Processing**: `numpy`, `pandas`, `scikit-learn`
- **Deep Learning**: `tensorflow`, `keras`
- **Visualization**: `matplotlib`
- **Web Scraping**: `requests`, `beautifulsoup4`
- **AI Integration**: `openai`
- **Financial Data**: `yfinance`
- **Optimization**: `scikit-optimize`
- **Technical Analysis**: `ta`
- **Utilities**: `tqdm`

## 🔧 Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd SentimentBayesLSTM
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

**Important**: You need to set up your OpenAI API key to use the sentiment analysis features.

1. Get your OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Replace the placeholder in the notebook:
   ```python
   api_key = 'YOUR_OPENAI_API_KEY_HERE'
   ```

## 📊 Usage

### Basic Usage

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook "SentimentBayesLSTM- Hybrid Forecasting with GPT Insights and Bayesian Tuning_VJ75 .ipynb"
   ```

2. Configure your parameters:
   ```python
   ticker = "AAPL"  # Stock symbol
   start_date = '2020-01-01'
   end_date = '2023-12-31'
   ```

3. Run all cells to:
   - Download stock data
   - Perform sentiment analysis
   - Train the LSTM model with Bayesian optimization
   - Generate predictions with uncertainty estimates

### Key Components

#### 1. StockPredictor Class
Base class for data downloading and preprocessing:
- Downloads historical stock data using yfinance
- Adds technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- Prepares data sequences for LSTM training

#### 2. SentimentAnalyzer Class
Handles news sentiment analysis:
- Scrapes Google News headlines
- Uses OpenAI GPT for sentiment scoring
- Caches results to minimize API calls
- Provides weekly sentiment aggregation

#### 3. LSTMModel Class
Advanced LSTM implementation:
- Multi-layer LSTM architecture with dropout
- Bayesian hyperparameter optimization
- Uncertainty estimation via Monte Carlo Dropout
- Comprehensive error handling

## 🎯 Model Architecture

### LSTM Network Structure
- **Input Layer**: Multi-feature time series data
- **LSTM Layers**: 3 stacked LSTM layers with dropout
- **Dense Output**: Single value prediction
- **Optimization**: Adam optimizer with tuned learning rate

### Features Used
- **Price Data**: Open, High, Low, Close, Volume
- **Technical Indicators**: SMA20, SMA50, EMA20, RSI, MACD, Bollinger Bands, ATR, OBV, KAMA, Stochastic, Williams %R
- **Sentiment Data**: Weekly aggregated news sentiment scores

### Hyperparameter Optimization
- **LSTM Units**: 64-256 (optimized)
- **Dropout Rate**: 0.1-0.5 (optimized)
- **Learning Rate**: 1e-4 to 1e-2 (log-uniform, optimized)
- **Optimization Method**: Gaussian Process (50 iterations)

## 📈 Performance Metrics

The model provides several evaluation metrics:
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **Prediction Uncertainty Bounds**
- **Visual Performance Plots**

## 🔒 Security Notes

- **Never commit API keys** to version control
- Use environment variables for sensitive data in production
- The provided notebook uses placeholder values for API keys
- Consider using API key management services for production deployments

## 📁 File Structure

```
├── SentimentBayesLSTM- Hybrid Forecasting with GPT Insights and Bayesian Tuning_VJ75 .ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── SentimentBayesLSTM_Full_User_Guide.pdf (detailed methodology guide)
└── cache files (generated during execution)
    ├── {ticker}_sentiment_cache.json
    └── model checkpoints
```

## 🚨 Important Notes

1. **API Costs**: OpenAI API calls incur costs. Monitor your usage.
2. **Rate Limits**: The code includes delays to respect API rate limits.
3. **Data Quality**: Ensure stable internet connection for data downloads.
4. **Computational Requirements**: Model training can be resource-intensive.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is provided for educational and research purposes. Please ensure compliance with all relevant APIs' terms of service.

## 🙏 Acknowledgments

- OpenAI for GPT API
- Yahoo Finance for stock data
- TensorFlow team for the deep learning framework
- scikit-optimize for Bayesian optimization tools

## 📞 Contact & Support

### For Academic Inquiries:
- **Full Dissertation Paper**: Contact me directly if you're interested in the complete academic research paper
- **Research Collaboration**: Open to discussing the methodology and findings
- **Academic Citations**: Please reach out for proper citation format

### For Technical Support:
1. Check the included PDF implementation guide
2. Review the notebook comments and documentation
3. Ensure all dependencies are properly installed
4. Verify API key configuration

### How to Reach Me:
Feel free to reach out through GitHub issues or direct contact for:
- Access to the full dissertation paper
- Questions about the research methodology
- Technical implementation details
- Potential collaborations

---

**Disclaimer**: This tool is for educational and research purposes only. Not financial advice. Always do your own research before making investment decisions.

**Academic Use**: If you use this work in your research, please provide appropriate attribution and consider reaching out for the full academic paper for comprehensive citations.
