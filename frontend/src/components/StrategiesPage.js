import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ScatterChart,
  Scatter,
  ComposedChart
} from "recharts";
import { motion } from "framer-motion";

const COLORS = ["#8884d8", "#82ca9d", "#ffc658", "#ff7f50", "#0088FE"];

function StrategiesPage() {
  const [ticker, setTicker] = useState("AAPL");
  const [strategyName, setStrategyName] = useState("BarUpDnStrategy");
  const [lookback, setLookback] = useState(252);
  const [initialCapital, setInitialCapital] = useState(10);
  const [maxLossPct, setMaxLossPct] = useState(2);
  const [strategyParams, setStrategyParams] = useState({});
  const [metrics, setMetrics] = useState(null);
  const [priceData, setPriceData] = useState([]);
  const [equityCurve, setEquityCurve] = useState([]);
  const [buyHoldCurve, setBuyHoldCurve] = useState([]);
  const [dates, setDates] = useState([]);
  const [trades, setTrades] = useState([]);
  const [plotBase64, setPlotBase64] = useState(null);
  const [performancePlot, setPerformancePlot] = useState(null);
  const [strategyParamsSchema, setStrategyParamsSchema] = useState([]);

  const [isRunning, setIsRunning] = useState(false);

  const windowOptions = [
    { label: "1W", days: 5 },
    { label: "1M", days: 252/12 },
    { label: "3M", days: 252/4 },
    { label: "6M", days: 252/2 },
    { label: "1Y", days: 252 },
    { label: "2Y", days: 504 },
    { label: "5Y", days: 252*5 },
    { label: "10Y", days: 252*10 },
  ];

  const STRATEGY_OPTIONS = [
    { label: "Bar Up Down Strategy", value: "BarUpDnStrategy" },
    { label: "Bollinger's Bands Strategy", value: "BollingerBandsStrategy" },
    { label: "Bollinger's Bands Strategy Directed", value: "BollingerDirectedStrategy"},
    { label: "Channel BreakOut Strategy", value: "ChannelBreakOutStrategy"},
    { label: "Consecutive Up/Down Strategy", value: "ConsecutiveUpDownStrategy"},
    { label: "Greedy Strategy", value: "GreedyStrategy"},
    { label: "Inside Bar Strategy", value: "InsideBarStrategy"},
    { label: "Keltner's Channels Strategy", value: "KeltnerChannelsStrategy"},
    { label: "MACD Strategy", value: "MACDStrategy"},
    { label: "Momentum Strategy", value: "MomentumStrategy" },
    { label: "MovingAvg2Line Cross Strategy", value: "MovingAvgCrossStrategy" },
    { label: "MovingAvg Cross Strategy", value: "MovingAvgCrossConfirmStrategy" },
    { label: "OutSide Bar", value: "OutSideBarStrategy" },
    { label: "Parabolic SAR", value: "ParabolicSARStrategy" },
    { label: "Pivot Extension Strategy", value: "PivotExtensionStrategy" },
    { label: "Price Channel Strategy", value: "PriceChannelStrategy" },
    { label: "ADX Breakout Strategy", value: "ADXBreakoutStrategy" },
    { label: "Stochastic Slow Strategy", value: "StochasticSlowStrategy" },
    { label: "Supertrend Strategy", value: "SupertrendStrategy" },



  ];

  const metricLabels = {
    final_equity: "Final Equity",
    total_pnl_usd: "Total PnL (USD)",
    total_pnl_pct: "Total PnL (%)",
    max_drawdown_usd: "Max Drawdown (USD)",
    max_drawdown_pct: "Max Drawdown (%)",
    total_trades: "Number of Trades",
    profitable_trades: "Profitable Trades",
    profit_factor: "Profit Factor",
    
  };
  
  
  useEffect(() => {
    if (!strategyName) return;
  
    axios
      .get(`https://portfoliomanegement-production-e103.up.railway.app/backtest/strategy_params/${strategyName}`)
      .then((res) => {
        setStrategyParamsSchema(res.data.params);
        
        // Initialiser les valeurs par défaut dans strategyParams
        const defaults = {};
        res.data.params.forEach(param => {
          defaults[param.name] = param.default ?? "";
        });
        setStrategyParams(defaults);
      })
      .catch(console.error);
  }, [strategyName]);
  
  

  const runBacktest = async () => {
    setIsRunning(true);
    const payload = {
      ticker: ticker.trim().toUpperCase(),
      strategy_name: strategyName,
      lookback_days: Number(lookback) || 252,
      strategy_params: Object.keys(strategyParams).length ? strategyParams : null,
      initial_capital: Number(initialCapital) || 100000,
      max_intraday_loss_pct: Number(maxLossPct) || 2,
    };

    try {
      const metricsRes = await axios.post(
      "https://portfoliomanegement-production-e103.up.railway.app/backtest/metrics",
      payload
    );

    const data = metricsRes.data;

    // ✅ Check if the backend returned an error
    if (data.error) {
      alert(data.error); // or use a styled modal instead of alert
      return; // stop further processing
    }
      const metricsData = metricsRes.data.metrics;

      setMetrics(metricsData)
      if (metricsData.profitable_trades === 0) {
        alert(
          "Warning: No profitable trades. Try adjusting the strategy parameters."
        );
      }

      const tradesRes = await axios.post("https://portfoliomanegement-production-e103.up.railway.app/backtest/trades", payload);
      setPriceData(tradesRes.data.price_data);
      setTrades(tradesRes.data.trades);

      const equityRes = await axios.post("https://portfoliomanegement-production-e103.up.railway.app/backtest/equity_curve", payload);
      setEquityCurve(equityRes.data.equity_curve);
      setDates(equityRes.data.dates);

      const bhRes = await axios.post("https://portfoliomanegement-production-e103.up.railway.app/backtest/buy_and_hold", payload);
      setBuyHoldCurve(bhRes.data.buy_and_hold);

      const plotRes = await axios.post("https://portfoliomanegement-production-e103.up.railway.app/backtest/plot", payload);
        setPlotBase64(plotRes.data.plot_base64);

        const response = await axios.post("https://portfoliomanegement-production-e103.up.railway.app/backtest/performance_plot", payload);
        setPerformancePlot(response.data.plot_base64);
        
      setIsRunning(false);
    } catch (err) {
      console.error(err.response?.data || err);
    }
  };

  return (
    <div style={{ padding: 32, fontFamily: "Segoe UI, sans-serif", background: "#f0f4f8", minHeight: "100vh" }}>
      <h1 style={{ textAlign: "center", marginBottom: 32 }}>Backtest Strategy</h1>
{/* ----- Top Inputs: 3 columns layout ----- */}
<div
  style={{
    display: "grid",
    gridTemplateColumns: "1fr 1fr 2fr", // Left, Middle, Right
    gap: 16,
    marginBottom: 32,
  }}
>
  {/* ----- Left: Ticker, Capital, Max Loss in ONE box ----- */}
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
  >
    <div
      style={{
        background: "white",
        borderRadius: 12,
        padding: 16,
        boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
        display: "grid",
        gap: 16,
      }}
    >
      {[
        { label: "Ticker", value: ticker, setter: setTicker, type: "text" },
        { label: "Initial Capital", value: initialCapital, setter: setInitialCapital, type: "number" },
        { label: "Max Intraday Loss (%)", value: maxLossPct, setter: setMaxLossPct, type: "number" },
      ].map((input) => (
        <div key={input.label}>
          <label style={{ fontWeight: 600 }}>{input.label}</label>
          <input
            type={input.type}
            value={input.value}
            onChange={(e) =>
              input.setter(input.type === "number" ? Number(e.target.value) : e.target.value)
            }
            style={{
              width: "100%",
              padding: 8,
              borderRadius: 6,
              border: "1px solid #cbd5e0",
              marginTop: 4
            }}
          />
        </div>
      ))}
    </div>
  </motion.div>

  {/* ----- Middle: Lookback ----- */}
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
  >
    <div
      style={{
        background: "white",
        borderRadius: 12,
        padding: 16,
        boxShadow: "0 2px 8px rgba(0,0,0,0.05)"
      }}
    >
      <label style={{ fontWeight: 600 }}>Lookback (days)</label>
      <input
        type="number"
        value={lookback}
        onChange={(e) => setLookback(Number(e.target.value))}
        style={{ width: "100%", padding: 8, borderRadius: 6, border: "1px solid #cbd5e0", marginTop: 8 }}
      />
      {/* Quick window buttons */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 8 }}>
        {windowOptions.map(({ label, days }) => (
          <button
            key={label}
            onClick={() => setLookback(days)}
            style={{
              padding: "4px 8px",
              backgroundColor: lookback === days ? "green" : "#2b6cb0",
              color: "white",
              border: "none",
              borderRadius: 6,
              cursor: "pointer",
            }}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  </motion.div>

  {/* ----- Right: Strategy + Parameters in ONE box ----- */}
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
  >
    <div
      style={{
        background: "white",
        borderRadius: 12,
        padding: 16,
        boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
        display: "grid",
        gap: 16,
      }}
    >
      {/* Strategy selector */}
      <div>
        <label style={{ fontWeight: 600 }}>Strategy</label>
        <select
          value={strategyName}
          onChange={(e) => setStrategyName(e.target.value)}
          style={{ width: "100%", padding: 8, borderRadius: 6, border: "1px solid #cbd5e0", marginTop: 4 }}
        >
          {STRATEGY_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      </div>

      {/* Strategy-specific parameters */}
      <div style={{ display: "grid", gap: 12 }}>
      {strategyParamsSchema.map((param) => (
  <div key={param.name}>
    <label style={{ fontWeight: 500 }}>{param.label}</label>

    {param.type === "boolean" ? (
      <input
        type="checkbox"
        checked={strategyParams[param.name] || false}
        onChange={(e) =>
          setStrategyParams((prev) => ({
            ...prev,
            [param.name]: e.target.checked
          }))
        }
        style={{ marginTop: 4 }}
      />
    ) : (
      <input
        type={param.type}
        value={strategyParams[param.name]}
        onChange={(e) =>
          setStrategyParams((prev) => ({
            ...prev,
            [param.name]: param.type === "number" ? Number(e.target.value) : e.target.value
          }))
        }
        style={{
          width: "100%",
          padding: 8,
          borderRadius: 6,
          border: "1px solid #cbd5e0",
          marginTop: 4
        }}
      />
    )}
  </div>
))}

      </div>
    </div>
  </motion.div>
</div>


      {/* ----- Run Backtest Button ----- */}
      <div style={{ textAlign: "center", marginBottom: 32 }}>
  <button
    onClick={runBacktest}
    disabled={isRunning} // Disable while running
    style={{
      padding: "12px 24px",
      backgroundColor: isRunning ? "#4a5568" : "#2b6cb0", // Gris quand en cours
      color: "white",
      border: "none",
      borderRadius: 8,
      cursor: isRunning ? "not-allowed" : "pointer",
      fontSize: 16,
      transition: "all 0.3s"
    }}
  >
    {isRunning ? "Running..." : "Run Backtest"}
  </button>
</div>


      {/* ----- Metrics ----- */}
        {metrics && (
        <>
            <h2 style={{ marginBottom: 16, fontSize: 24, fontWeight: 600 }}>
            Metrics
            </h2>

            <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                gap: 16,
                marginBottom: 32
            }}
            >
            {Object.entries(metrics).map(([key, value]) => (
                <div
                key={key}
                style={{
                    background: "#f7fafc",
                    padding: 16,
                    borderRadius: 12,
                    boxShadow: "0 1px 4px rgba(0,0,0,0.05)"
                }}
                >
                <strong style={{ fontSize: 15 }}>
                    {metricLabels[key] || key}
                </strong>

                <div style={{ fontSize: 18, marginTop: 4 }}>
                    {typeof value === "number" ? value.toFixed(2) : value}
                </div>
                </div>
            ))}
            </motion.div>
        </>
        )}


      {/* ----- Charts side by side ----- */}
<div
  style={{
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 24,
    marginTop: 32,
  }}
>
  {/* ----- Equity Curve Chart ----- */}
  {performancePlot && (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.4 }}
  >
    <h2 className="text-xl font-semibold mb-4">
      Strategy vs Buy & Hold
    </h2>

    <img
      src={`data:image/png;base64,${performancePlot}`}
      alt="Performance Plot"
      style={{
        width: "100%",
        height: "auto",
        maxHeight: 500,
        borderRadius: "12px",
        boxShadow: "0 4px 20px rgba(0,0,0,0.15)",
      }}
    />
  </motion.div>
)}


  {/* ----- Price Chart with Trades (Strategy Plot) ----- */}
{plotBase64 && (
  <motion.div 
  initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.4 }}>
    <h2 className="text-xl font-semibold mb-4">Price Chart with Trades</h2>
    {/* Image renvoyée par le backend */}
    <img
      src={`data:image/png;base64,${plotBase64}`}
      alt="Strategy Plot"
      style={{
        width: "100%",
        height: "auto",
        maxHeight: 500,
        borderRadius: "12px",
        boxShadow: "0 4px 20px rgba(0,0,0,0.15)",
      }}
    />
  </motion.div>
)}

</div>

    </div>
  );
}

export default StrategiesPage;
