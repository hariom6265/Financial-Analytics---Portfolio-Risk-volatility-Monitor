<body>
<div class="container">
  
  <h1>📋 Financial Analytics — Portfolio Risk &amp; Volatility Monitor</h1>
  

  <h2>📝 Summary</h2>
  <div class="card">
    <p>A Python-based financial analytics system that monitors portfolio risk through volatility analysis, Monte Carlo simulations, and real-time asset pricing using live market data. The project fetches historical data from multiple asset classes, calculates risk metrics, and provides visual dashboards for portfolio performance tracking.</p>
  </div>
  
  <h2>🎯 Overview</h2>
  <ul>
    <li><strong>Week 1:</strong> Data collection and portfolio volatility calculation</li>
    <li><strong>Week 2:</strong> Monte Carlo simulation for Value at Risk (VaR) estimation</li>
    <li><strong>Week 3:</strong> Sector-based portfolio analysis and allocation</li>
  </ul>
  <p>The system tracks multiple asset classes (Stocks, ETFs, Bitcoin) and provides real-time risk metrics and visualization.</p>

  <h2>⚠️ Problem Statement</h2>
  <ul>
    <li>Monitor portfolio volatility in real-time</li>
    <li>Estimate downside risk using Value at Risk (VaR)</li>
    <li>Track asset allocation across sectors</li>
    <li>Identify anomalies and outliers in price movements</li>
    <li>Project future portfolio performance through simulations</li>
  </ul>

  <h2>📊 Dataset</h2>
  <ul>
    <li><strong>Sources:</strong> Yahoo Finance (<code>yfinance</code> library)</li>
    <li><strong>Tickers:</strong> AAPL, JNJ, XOM, SPY, BTC-USD</li>
    <li><strong>Time Period:</strong> January 1, 2023 – January 1, 2025</li>
  </ul>

  <h3>✅ Data Types</h3>
  <ul>
    <li>Historical price data (Open, High, Low, Close, Volume)</li>
    <li>Log returns</li>
    <li>Closing prices</li>
  </ul>

  <h3>✅ Output Files</h3>
  <ul>
    <li><code>cleaned_price_data.csv</code> — Processed price data</li>
    <li><code>log_returns.csv</code> — Calculated log returns</li>
    <li><code>portfolio_data.csv</code> — Sector allocation data</li>
  </ul>

  <h2>🛠️ Tools &amp; Technologies</h2>
  <table>
    <tr><th>Category</th><th>Tools</th></tr>
    <tr><td>Data Fetching</td><td>yfinance</td></tr>
    <tr><td>Data Processing</td><td>Pandas, NumPy</td></tr>
    <tr><td>Analysis</td><td>NumPy (Statistical calculations)</td></tr>
    <tr><td>Visualization</td><td>Matplotlib</td></tr>
    <tr><td>Language</td><td>Python 3.x</td></tr>
    <tr><td>Libraries</td><td>scipy, numpy, pandas, matplotlib</td></tr>
  </table>

  <h2>🔬 Methods</h2>

  <span class="week-tag">Week 1</span>
  <h3>Portfolio Volatility Calculation</h3>
  <ol>
    <li>Data Retrieval → <code>yfinance.download()</code></li>
    <li>Forward/Backward Fill → Handle missing values</li>
    <li>Log Returns Calculation → ln(Price_t / Price_t-1)</li>
    <li>Mean &amp; Covariance Matrix → Statistical analysis</li>
    <li>Equal-Weight Portfolio → weights = 1/n</li>
    <li>Portfolio Return = E(w<sup>T</sup> × μ)</li>
    <li>Portfolio Volatility = √(w<sup>T</sup> × Σ × w)</li>
    <li>Outlier Detection → Z-score filtering (|z| &gt; 5)</li>
  </ol>

  <span class="week-tag">Week 2</span>
  <h3>Monte Carlo Simulation</h3>
  <p><strong>Geometric Brownian Motion (GBM):</strong></p>
  <div class="formula">S_t = S_0 × exp((μ − σ²/2)×dt + σ×√dt×Z_t)</div>
  <p><strong>Parameters:</strong></p>
  <ul>
    <li>μ (drift) = mean daily return</li>
    <li>σ (volatility) = std dev of returns</li>
    <li>N = 252 trading days</li>
    <li>Simulations = 10,000 paths</li>
  </ul>
  <p><strong>VaR Calculation:</strong></p>
  <ul>
    <li>95% Confidence Level (5th percentile)</li>
    <li>Maximum expected loss</li>
  </ul>

  <span class="week-tag">Week 3</span>
  <h3>Portfolio Allocation</h3>
  <ol>
    <li>Sector Mapping</li>
    <li>Weight Distribution</li>
    <li>Asset Concentration Analysis</li>
  </ol>

  <h2>💡 Key Insights</h2>
  <table>
    <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
    <tr><td>Portfolio Volatility</td><td>0.97%</td><td>Daily volatility ~1%</td></tr>
    <tr><td>Average Daily Return</td><td>0.06%</td><td>Modest daily gains</td></tr>
    <tr><td>Total Portfolio Value</td><td>107.92K</td><td>Current allocation</td></tr>
    <tr><td>VaR (95%)</td><td>Calculated</td><td>Max 5% daily loss probability</td></tr>
    <tr><td>Asset-wise Volatility</td><td>0.69%–1.07%</td><td>Mixed risk profile</td></tr>
  </table>

  <h3>✅ Key Findings</h3>
  <div class="metrics-grid">
    <div class="metric-item">📉 Portfolio maintains moderate daily volatility (~0.97%)</div>
    <div class="metric-item">🧩 Diversification across 5 assets reduces concentration risk</div>
    <div class="metric-item">📱 Tech stocks (AAPL) show higher volatility (1.07%)</div>
    <div class="metric-item">₿ Bitcoin exhibits highest volatility among assets</div>
    <div class="metric-item">⚖️ Equal-weight portfolio provides baseline diversification</div>
  </div>

  <h2>📊 Dashboard Overview</h2>
  <img class="dashboard-img" width="800" height="448" alt="Financial Analytics - Portfolio Risk & Volatility Monitor Dashboard" src="https://github.com/user-attachments/assets/eb52db5f-5116-47f3-8cb8-121b10dc923a" />

  <h3>✅ KPI Cards (Top Section)</h3>
  <ul>
  <li><strong>Total Portfolio Value:</strong> 107.92K</li>
  <li><strong>Average Daily Return:</strong> 0.06%</li>
  <li><strong>Portfolio Volatility:</strong> 0.97%</li>
  <li><strong>Portfolio CAGR:</strong> 3.01K</li>
</ul>

  <h3>Asset Summary Table</h3>
  <table>
    <tr><th>Asset</th><th>Ticker</th><th>Avg Price</th><th>Volatility</th><th>Return</th></tr>
    <tr><td>ICICI Presidential Gold ETF</td><td>ICIGOLD</td><td>1,797.38</td><td>1.85%</td><td>0.12%</td></tr>
    <tr><td>SBI ETF Nifty 50</td><td>SBIEFT50</td><td>601.79</td><td>0.81%</td><td>0.11%</td></tr>
    <tr><td>Infosys</td><td>INFY</td><td>1,472.63</td><td>0.96%</td><td>0.10%</td></tr>
    <tr><td>Axis Bluechip Fund</td><td>AXISBLUCHIP</td><td>293.28</td><td>1.07%</td><td>0.04%</td></tr>
    <tr><td>HDFC Bank</td><td>HDFCBANK</td><td>1,585.83</td><td>0.93%</td><td>-0.05%</td></tr>
    <tr><td><strong>Total</strong></td><td></td><td></td><td><strong>0.97%</strong></td><td><strong>0.06%</strong></td></tr>
  </table>

  <h3>✅ Visualizations</h3>
  <ul>
    <li><strong>Asset Allocation (Pie Chart):</strong> Sector distribution</li>
    <li><strong>Time Series:</strong> Price movements over time</li>
    <li><strong>Asset-wise Volatility (Bar Chart):</strong> Risk comparison</li>
  </ul>

  <h2>🔧 How to Run This</h2>

  <p><span class="step-num">Step 1:</span> Clone the Repository</p>
 <pre><code>git clone <a href="https://github.com/dna5421/Financial-Analytics---Portfolio-Risk-volatility-Monitor.git" target="_blank">https://github.com/dna5421/Financial-Analytics---Portfolio-Risk-volatility-Monitor.git</a>
cd Financial-Analytics---Portfolio-Risk-volatility-Monitor</code></pre>
  <p><span class="step-num">Step 2:</span> Install Dependencies</p>
  <pre><code>pip install yfinance numpy pandas matplotlib scipy</code></pre>

  <p><span class="step-num">Step 3:</span> Run Individual Scripts</p>

  <p><strong>Week 1 — Volatility Analysis</strong></p>
  <pre><code>python week1.py</code></pre>
  <p>✅ Output:</p>
  <ul>
    <li><code>cleaned_price_data.csv</code></li>
    <li><code>log_returns.csv</code></li>
    <li>Console: Expected Daily Return, Daily Volatility</li>
  </ul>

  <p><strong>Week 2 — Monte Carlo Simulation</strong></p>
  <pre><code>python week2.py</code></pre>
  <p>✅ Output:</p>
  <ul>
    <li>Console: Annualized Volatility, VaR (95%)</li>
    <li>Visualization: Monte Carlo price paths</li>
  </ul>

  <p><strong>Week 3 — Portfolio Allocation</strong></p>
  <pre><code>python week3.py</code></pre>
  <p>Output: <code>portfolio_data.csv</code> with sector allocation</p>

  <p><span class="step-num">Step 4:</span> View Results</p>
  <pre><code># Check generated CSV files
cat cleaned_price_data.csv
cat log_returns.csv
cat portfolio_data.csv</code></pre>

  <p><span class="step-num">Step 5:</span> (Optional) Custom Analysis</p>
  <p>Modify the TICKERS and date range in <code>week1.py</code> for different portfolios:</p>
  <pre><code>TICKERS = ["TICKER1", "TICKER2", "TICKER3"]  # Your assets
START_DATE = "YYYY-MM-DD"
END_DATE = "YYYY-MM-DD"</code></pre>

  <h2>📈 Results</h2>

  <h3>✅ Quantitative Results</h3>
  <div class="result-item">1. <strong>Portfolio Daily Volatility:</strong> 0.97% · Annualized: ~15.4% (0.97% × √252)</div>
  <div class="result-item">2. <strong>Daily Returns:</strong> 0.06% · Annualized: ~15% (0.06% × 252)</div>
  <div class="result-item">3. <strong>Value at Risk (95% confidence):</strong> 5th percentile of return distribution — maximum expected loss with 95% confidence</div>
  <div class="result-item">4. <strong>Monte Carlo Insights:</strong> 10,000 simulation paths generated, captures fat tails and extreme events, projects year-ahead price distributions</div>
  <div class="result-item">5. <strong>Asset Volatility Rankings:</strong> Highest — ICIGOLD (1.85%), AXISBLUCHIP (1.07%); Lowest — SBIEFT50 (0.81%), HDFCBANK (0.93%)</div>

  <h2>✅ Conclusion</h2>

  <h3>1. Project Success</h3>
  <ul>
    <li>✓ Successfully fetches and processes multi-asset financial data</li>
    <li>✓ Calculates comprehensive risk metrics (volatility, VaR)</li>
    <li>✓ Implements industry-standard Monte Carlo simulations</li>
    <li>✓ Provides visual dashboard for portfolio monitoring</li>
    <li>✓ Handles data quality through forward/backward fill and outlier detection</li>
  </ul>

  <h3>2. Risk Profile Summary</h3>
  <ul>
    <li><strong>Portfolio Type:</strong> Moderate-risk, diversified</li>
    <li><strong>Risk Level:</strong> ~0.97% daily volatility (moderate)</li>
    <li><strong>Diversification:</strong> 5 assets across sectors</li>
    <li><strong>Time Horizon:</strong> Suitable for medium-term investors</li>
  </ul>

  <h3>3. Strengths</h3>
  <ul>
    <li>Real-time data sourcing</li>
    <li>Robust error handling and retries</li>
    <li>Multi-asset class support</li>
    <li>Clear risk metrics</li>
  </ul>

  <h3>4. Limitations</h3>
  <div class="limitation-item">⚠️ Equal-weight allocation (not optimized)</div>
  <div class="limitation-item">⚠️ Historical analysis (backward-looking)</div>
  <div class="limitation-item">⚠️ Assumes lognormal distribution</div>
  <div class="limitation-item">⚠️ No transaction costs included</div>

  <h2>🚀 Future Work</h2>

  <h3>✅ Short-term Enhancements</h3>
  <p><strong>1. Portfolio Optimization</strong></p>
  <ul>
    <li>Implement Markowitz efficient frontier</li>
    <li>Calculate optimal asset weights</li>
    <li>Minimize volatility for target returns</li>
  </ul>

  <p><strong>2. Advanced Risk Metrics</strong></p>
  <ul>
    <li>Conditional VaR (CVaR)</li>
    <li>Expected Shortfall (ES)</li>
    <li>Sharpe Ratio optimization</li>
  </ul>

  <p><strong>3. Real-time Dashboard</strong></p>
  <ul>
    <li>Streamlit/Dash web interface</li>
    <li>Live data updates</li>
    <li>Interactive parameter adjustment</li>
  </ul>

  <h3>✅ Medium-term Additions</h3>
  <p><strong>4. Correlation Analysis</strong></p>
  <ul>
    <li>Correlation matrix visualization</li>
    <li>Principal Component Analysis (PCA)</li>
    <li>Risk decomposition</li>
  </ul>

  <p><strong>5. Stress Testing</strong></p>
  <ul>
    <li>Scenario analysis</li>
    <li>Historical stress scenarios</li>
    <li>Sensitivity analysis</li>
  </ul>

  <p><strong>6. Machine Learning Integration</strong></p>
  <ul>
    <li>Price prediction models</li>
    <li>Anomaly detection</li>
    <li>Pattern recognition</li>
  </ul>

  <h3>✅ Long-term Vision</h3>
  <p><strong>7. Multi-factor Models</strong></p>
  <ul>
    <li>Fama-French factors</li>
    <li>Risk factor analysis</li>
    <li>Attribution analysis</li>
  </ul>

  <p><strong>8. Portfolio Management Features</strong></p>
  <ul>
    <li>Rebalancing strategies</li>
    <li>Backtesting framework</li>
    <li>Performance attribution</li>
  </ul>

  <p><strong>9. Enhanced Visualization</strong></p>
  <ul>
    <li>3D portfolio surface plots</li>
    <li>Heatmaps for correlations</li>
    <li>Risk decomposition charts</li>
  </ul>

  <h2>👤 Author &amp; Contact</h2>
  <ul>
    <li>👤 Author: <strong>dna5421</strong></li>
    <li>📧 GitHub Profile: <a href="https://github.com/dna5421">github.com/dna5421</a></li>
    <li>🔗 Repository: <a href="https://github.com/dna5421/Financial-Analytics---Portfolio-Risk-volatility-Monitor">github.com/dna5421/Financial-Analytics---Portfolio-Risk-volatility-Monitor</a></li>
  </ul>

  <hr>
  <div class="footer">
    <p>Built with Python, yfinance &amp; Monte Carlo Simulation · Financial Analytics Project</p>
  </div>

</div>
</body>
</html>
