import React, { useState, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  PieChart, Pie, Cell, ResponsiveContainer
} from "recharts";
import axios from "axios";
import { motion } from "framer-motion";
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';


const COLORS = ["#8884d8", "#82ca9d", "#ffc658", "#ff7f50", "#0088FE"];

const defaultPortfolio = {
  start: "2021-01-01",
  end: new Date().toISOString().slice(0, 10),
  tickers: ["AIR.PA", "BA", "TSLA", "AAPL"],
  positions: [40, 20, 20, 30],
  window: 365
};

const defaultMomentumParams = { short_window: 5, long_window: 30 };
const defaultConstVolParams = { target_vol: 0.2, vol_lookback: 10 };

function PortfolioDashboard() {
  const [portfolio, setPortfolio] = useState(defaultPortfolio);
  const [momentumParams, setMomentumParams] = useState(defaultMomentumParams);
  const [constVolParams, setConstVolParams] = useState(defaultConstVolParams);
  const [openStrategyModal, setOpenStrategyModal] = useState(null); // null, "momentum" ou "constvol"
  const [strategyWindow, setStrategyWindow] = useState(portfolio.window);


  const [metrics, setMetrics] = useState(null);
  const [returns, setReturns] = useState(null);
  const [pieData, setPieData] = useState([]);
  const [strategies, setStrategies] = useState(null);

  const [loadingMetrics, setLoadingMetrics] = useState(false);
  const [loadingBacktest, setLoadingBacktest] = useState(false);

  // ---- Fetch Portfolio Metrics ----
  const fetchMetrics = async () => {
    setLoadingMetrics(true);
    try {
      const res = await axios.post("https://portfoliomanegement-production.up.railway.app/metrics", portfolio);
      setMetrics(res.data);

      if (res.data.portfolio_values && res.data.dates) {
        const normalized = res.data.portfolio_values.map((v, i) => ({
          date: res.data.dates[i],
          value: v / res.data.portfolio_values[0],
        }));
        setReturns(normalized);
      }

      if (res.data.latest_prices) {
        const prices = res.data.latest_prices;
        const totalValue = portfolio.positions.reduce(
          (acc, pos, i) => acc + pos * prices[i],
          0
        );
        const pie = portfolio.tickers.map((t, i) => ({
          name: t,
          value: ((portfolio.positions[i] * prices[i]) / totalValue) * 100,
        }));
        setPieData(pie);
      }
    } catch (err) {
      console.error("❌ Error fetching metrics:", err.message);
    } finally {
      setLoadingMetrics(false);
    }
  };

  const computePieData = async () => {
    try {
      const res = await axios.post("https://portfoliomanegement-production.up.railway.app/metrics", portfolio);
      if (res.data.latest_prices) {
        const prices = res.data.latest_prices;
        const totalValue = portfolio.positions.reduce(
          (acc, pos, i) => acc + pos * prices[i],
          0
        );
        const pie = portfolio.tickers.map((t, i) => ({
          name: t,
          value: ((portfolio.positions[i] * prices[i]) / totalValue) * 100
        }));
        setPieData(pie);
      }
    } catch (err) {
      console.error("Error computing pie data:", err);
    }
  };
  

  // ---- Run Backtest ----
  const fetchStrategies = async () => {
    setLoadingBacktest(true);
    try {
      // Calcul de la date de début en fonction de strategyWindow
      const endDate = new Date(portfolio.end);
      const startDate = new Date(endDate);
      startDate.setDate(endDate.getDate() - strategyWindow); // recule de strategyWindow jours
  
      const payload = {
        portfolio,
        momentumParams,
        constVolParams,
        start: startDate.toISOString().slice(0, 10),
        end: portfolio.end,
      };
  
      const res = await axios.post("https://portfoliomanegement-production.up.railway.app/backtest", payload);
      setStrategies(res.data);
    } catch (err) {
      console.error("❌ Error fetching strategies:", err.message);
    } finally {
      setLoadingBacktest(false);
    }
  };
  

  // ---- Input Handlers ----
  const handlePortfolioChange = (e) => {
    const { name, value } = e.target;
    if (name === "tickers") {
      setPortfolio(prev => ({ ...prev, tickers: value.split(",").map(v => v.trim()) }));
    } else if (name === "positions") {
      setPortfolio(prev => ({ ...prev, positions: value.split(",").map(v => Number(v.trim())) }));
    } else if (name === "window") {
      setPortfolio(prev => ({ ...prev, window: Number(value) }));
    } else {
      setPortfolio(prev => ({ ...prev, [name]: value }));
    }
  };

  const handleMomentumChange = (e) => {
    const { name, value } = e.target;
    setMomentumParams(prev => ({ ...prev, [name]: Number(value) }));
  };

  const handleConstVolChange = (e) => {
    const { name, value } = e.target;
    setConstVolParams(prev => ({ ...prev, [name]: Number(value) }));
  };

  const METRIC_LABELS = {
    CAGR: "Annualized Return CAGR (%)",
    AnnualVolatility: "Annualized Volatility (%)",
    SharpeRatio: "Sharpe Ratio (Annual)",
    MaxDrawdown: "Max Drawdown (%)",
    VaR_Historical: "Historical VaR (at 95%)",
    CVaR_Historical: "Historical CVaR (at 95%)",
  };

  const METRIC_DECIMALS = {
    CAGR: 2,
    AnnualVolatility: 2,
    SharpeRatio: 2,
    MaxDrawdown: 2,
    VaR_Historical: 0,
    CVaR_Historical: 0,
  };

  const PERCENT_METRICS = ["CAGR", "AnnualVolatility", "MaxDrawdown", "VaR_Historical", "CVaR_Historical"];

  const [portfolioPrices, setPortfolioPrices] = useState([]);

  async function fetchPrices() {
    const res = await fetch("https://portfoliomanegement-production.up.railway.app/prices", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(portfolio),
    });
    const json = await res.json();
    setPortfolioPrices(json.data);
  }

  useEffect(() => {
    if (portfolio.tickers.every(t => t.trim() !== "")) {
      fetchPrices();
    }
  }, [portfolio.window, portfolio.tickers]);
  

  const lineChartData = portfolioPrices.map((row) => {
    const obj = { date: row.date };
    portfolio.tickers.forEach((ticker, i) => {
      obj[ticker] = row.prices[i];
    });
    return obj;
  });


  // ---- UI ----
  return (
    <div
      style={{
        padding: 32,
        fontFamily: "Segoe UI, sans-serif",
        background: "#f0f4f8",
        minHeight: "100vh",
      }}
    >
      <h1 style={{ textAlign: "center", fontSize: 32, marginBottom: 32 }}>
        Portfolio Management Dashboard
      </h1>

      {/* ---- Portfolio Definition ---- */}
      <motion.div
  initial={{ opacity: 0, y: 20 }}
  animate={{ opacity: 1, y: 0 }}
  style={{
    backgroundColor: "white",
    borderRadius: 12,
    padding: 24,
    marginBottom: 32,
    boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
  }}
>
  <h2>Portfolio Definition</h2>

  <div style={{ display: "flex", gap: 32, flexWrap: "wrap" }}>
  {/* ----- Tableau Tickers / Positions ----- */}
  <div
    style={{
      flex: 2,
      minWidth: 300,
      border: "1px solid #cbd5e0",
      borderRadius: 8,
      padding: 16,
      backgroundColor: "#f8fafc",
    }}
  >
    <table style={{ width: "100%", borderCollapse: "collapse", marginBottom: 16 }}>
      <thead>
        <tr>
          <th style={{ textAlign: "left", padding: 8 }}>Ticker</th>
          <th style={{ textAlign: "left", padding: 8 }}>Position</th>
          <th style={{ textAlign: "center", padding: 8 }}>Action</th>
        </tr>
      </thead>
      <tbody>
        {portfolio.tickers.map((ticker, idx) => (
          <tr key={idx} style={{ borderBottom: "1px solid #e2e8f0" }}>
            <td style={{ padding: 8 }}>
              <input
                type="text"
                value={ticker}
                onChange={(e) => {
                  const newTickers = [...portfolio.tickers];
                  newTickers[idx] = e.target.value;
                  setPortfolio(prev => ({ ...prev, tickers: newTickers }));
                }}
                style={{
                  width: "100%",
                  padding: "8px 10px",
                  borderRadius: 6,
                  border: "1px solid #cbd5e0",
                  marginRight: 8, // espace entre les inputs
                }}
              />
            </td>
            <td style={{ padding: 8 }}>
              <input
                type="number"
                value={portfolio.positions[idx]}
                onChange={(e) => {
                  const newPositions = [...portfolio.positions];
                  newPositions[idx] = Number(e.target.value);
                  setPortfolio(prev => ({ ...prev, positions: newPositions }));
                }}
                style={{
                  width: "100%",
                  padding: "8px 10px",
                  borderRadius: 6,
                  border: "1px solid #cbd5e0",
                  marginLeft: 8, // espace entre les inputs
                }}
              />
            </td>
            <td style={{ padding: 8, textAlign: "center" }}>
              <button
                onClick={() => {
                  const newTickers = portfolio.tickers.filter((_, i) => i !== idx);
                  const newPositions = portfolio.positions.filter((_, i) => i !== idx);
                  setPortfolio(prev => ({ ...prev, tickers: newTickers, positions: newPositions }));
                }}
                style={{
                  padding: "4px 8px",
                  backgroundColor: "#e53e3e",
                  color: "white",
                  border: "none",
                  borderRadius: 6,
                  cursor: "pointer",
                }}
              >
                Delete
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>

    <button
      onClick={() => {
        setPortfolio(prev => ({
          ...prev,
          tickers: [...prev.tickers, ""],
          positions: [...prev.positions, 0],
        }));
      }}
      style={{
        padding: "6px 12px",
        backgroundColor: "#2b6cb0",
        color: "white",
        border: "none",
        borderRadius: 6,
        cursor: "pointer",
      }}
    >
    Add a ticker
    </button>
  </div>


    {/* ----- Input Window with Quick Buttons ----- */}
<div style={{ flex: 1, minWidth: 220, display: "flex", flexDirection: "column", gap: 16 }}>
  <div>
    <label style={{ fontWeight: 600 }}>Window (days)</label>
    <input
      type="number"
      name="window"
      value={portfolio.window}
      onChange={handlePortfolioChange}
      style={{
        width: "100%",
        padding: "10px 12px",
        borderRadius: 8,
        border: "1px solid #cbd5e0",
        fontSize: 14,
        marginBottom: 8
      }}
    />
  </div>
{/* Quick selection buttons */}
<div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
  {[
    { label: "1W", days: 7 },
    { label: "1M", days: 30 },
    { label: "3M", days: 90 },
    { label: "6M", days: 180 },
    { label: "1Y", days: 365 },
    { label: "2Y", days: 730 },
    { label: "5Y", days: 1825 },
    { label: "10Y", days: 3650 },
  ].map(({ label, days }) => {
    const isSelected = portfolio.window === days; // window est number
    return (
      <button
        key={label}
        onClick={() => setPortfolio(prev => ({ ...prev, window: days }))}
        style={{
          padding: "6px 12px",
          backgroundColor: isSelected ? "green" : "#2b6cb0", // vert si sélectionné
          color: "white",
          border: "none",
          borderRadius: 6,
          cursor: "pointer",
          fontSize: 14
        }}
      >
        {label}
      </button>
    );
  })}
</div>

  {/* ----- Bouton Update Metrics ----- */}
  <button
    onClick={async () => {
      setLoadingMetrics(true);
      await fetchMetrics();
      await computePieData();
      setLoadingMetrics(false);
    }}
    style={{
      marginTop: 24,
      padding: "12px 24px",
      backgroundColor: "#2b6cb0",
      color: "white",
      border: "none",
      borderRadius: 8,
      cursor: "pointer",
      fontSize: 16
    }}
  >
    {loadingMetrics ? "Updating..." : "Update Metrics"}
  </button>

  </div>

</div>
</motion.div>



      {/* ---- Metrics + Charts ---- */}
      {metrics && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          style={{
            backgroundColor: "white",
            borderRadius: 12,
            padding: 24,
            marginBottom: 32,
            boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
          }}
        >
          <h2>Portfolio Metrics</h2>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
            {Object.keys(METRIC_LABELS).map((key) =>
              metrics[key] !== undefined && (
                <div
                  key={key}
                  style={{
                    flex: "1 1 150px",
                    backgroundColor: "#f7fafc",
                    padding: 12,
                    borderRadius: 8,
                  }}
                >
                  <strong>{METRIC_LABELS[key]}</strong>
                  <div style={{ fontSize: 18, marginTop: 4 }}>
                    {typeof metrics[key] === "number"
                      ? metrics[key].toFixed(key === "SharpeRatio" ? 2 : 0) + (PERCENT_METRICS.includes(key) ? "%" : "")
                      : metrics[key]}
                  </div>
                </div>
              )
            )}
          </div>




          {/* ---- Charts Side by Side ---- */}
          {returns && pieData.length > 0 && (
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "flex-start",
                gap: 32,
                marginTop: 24,
              }}
            >
              <div style={{ flex: 2 }}>
                <h3>Cumulative Return</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={returns}>
                    <CartesianGrid stroke="#ccc" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line
                      type="linear"
                      dataKey="value"
                      stroke="#2b6cb0"
                      dot={false}
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div style={{ flex: 1 }}>
  <h3>Capital Allocation (%)</h3>
  <ResponsiveContainer width="100%" height={300}>
    <PieChart>
      <Pie
        data={pieData}
        dataKey="value"
        nameKey="name"
        cx="50%"
        cy="50%"
        outerRadius={100}
        label={({ name, value }) => `${name}: ${Math.round(value)}%`} // arrondi sans décimale
      >
        {pieData.map((entry, i) => (
          <Cell key={i} fill={COLORS[i % COLORS.length]} />
        ))}
      </Pie>
      <Tooltip formatter={(value) => `${Math.round(value)}%`} /> 
      <Legend
        verticalAlign="bottom"
        height={36}
        formatter={(value, entry) => {
          // Cherche l'index du ticker dans pieData pour récupérer la couleur correcte
          const index = pieData.findIndex(d => d.name === value);
          const color = COLORS[index % COLORS.length];
          return <span style={{ color }}>{value}</span>;
        }}
      />
    </PieChart>
  </ResponsiveContainer>
</div>


            </div>
          )}
          

<div style={{ flex: 1, marginTop: 32 }}>
  <h3>Stock Prices Over Window</h3>
  <ResponsiveContainer width="100%" height={400}>
  <LineChart data={lineChartData}>
    <XAxis dataKey="date" />
    <YAxis />
    <Tooltip />
    <Legend />
    {portfolio.tickers.map((ticker, i) => (
      <Line
        key={ticker}
        type="monotone"
        dataKey={ticker}
        stroke={COLORS[i % COLORS.length]}
        dot={false}
      />
    ))}
  </LineChart>
</ResponsiveContainer>

</div>
        </motion.div>
      )}

      {/* ---- Backtested Strategies ---- */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        style={{
          backgroundColor: "white",
          borderRadius: 12,
          padding: 24,
          boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
        }}
      >
        <h2>Backtested Strategies</h2>
        <div style={{ display: "flex", gap: 24, flexWrap: "wrap", marginBottom: 16 }}>
          <div
            style={{
              flex: 1,
              background: "#f9fafb",
              padding: 16,
              borderRadius: 8,
              border: "1px solid #e2e8f0",
            }}
          >
            <h4>Momentum Strategy
            <button
          onClick={() => setOpenStrategyModal("momentum")}
          style={{
            marginLeft: 8,
            borderRadius: "50%",
            width: 20,
            height: 20,
            border: "none",
            backgroundColor: "#2b6cb0",
            color: "white",
            cursor: "pointer",
            fontWeight: "bold",
          }}
        >
          ?
        </button>
            </h4>
            <label>Short Window</label>
            <input
              type="number"
              name="short_window"
              value={momentumParams.short_window}
              onChange={handleMomentumChange}
              style={{ width: 80, padding: 4, marginRight: 8 }}
            />
            <label>Long Window</label>
            <input
              type="number"
              name="long_window"
              value={momentumParams.long_window}
              onChange={handleMomentumChange}
              style={{ width: 80, padding: 4 }}
            />
          </div>

          <div
            style={{
              flex: 1,
              background: "#f9fafb",
              padding: 16,
              borderRadius: 8,
              border: "1px solid #e2e8f0",
            }}
          >
            <h4>Constant Volatility
            <button
          onClick={() => setOpenStrategyModal("constvol")}
          style={{
            marginLeft: 8,
            borderRadius: "50%",
            width: 20,
            height: 20,
            border: "none",
            backgroundColor: "#2b6cb0",
            color: "white",
            cursor: "pointer",
            fontWeight: "bold",
          }}
        >
          ?
        </button>
            </h4>
            <label>Target Vol</label>
            <input
              type="number"
              step="0.01"
              name="target_vol"
              value={constVolParams.target_vol}
              onChange={handleConstVolChange}
              style={{ width: 80, padding: 4, marginRight: 8 }}
            />
            <label>Vol Lookback</label>
            <input
              type="number"
              name="vol_lookback"
              value={constVolParams.vol_lookback}
              onChange={handleConstVolChange}
              style={{ width: 80, padding: 4 }}
            />
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 12, marginBottom: 16 }}>
  {/* Titre */}
  <label style={{ fontWeight: 600 }}>Strategy Window (days)</label>

  {/* Input numérique */}
  <input
    type="number"
    value={strategyWindow}
    onChange={(e) => setStrategyWindow(Number(e.target.value))}
    style={{
      width: 80,
      padding: "6px 8px",
      borderRadius: 6,
      border: "1px solid #cbd5e0",
      fontSize: 14,
    }}
  />

  {/* Quick selection buttons */}
  <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
    {[
      { label: "1W", days: 7 },
      { label: "1M", days: 30 },
      { label: "3M", days: 90 },
      { label: "6M", days: 180 },
      { label: "1Y", days: 365 },
      { label: "2Y", days: 730 },
      { label: "5Y", days: 1825 },
      { label: "10Y", days: 3650 },
    ].map(({ label, days }) => (
      <button
        key={label}
        onClick={() => setStrategyWindow(days)}
        style={{
          padding: "4px 8px",
          backgroundColor: strategyWindow === days ? "green" : "#2b6cb0",
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
</div>


  
{/* ---- Modal pour l'explication ---- */}
{openStrategyModal && (
  <div
    onClick={() => setOpenStrategyModal(null)}
    style={{
      position: "fixed",
      top: 0, left: 0,
      width: "100%", height: "100%",
      backgroundColor: "rgba(0,0,0,0.5)",
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      zIndex: 1000,
    }}
  >
    <div
      onClick={(e) => e.stopPropagation()}
      style={{
        backgroundColor: "white",
        padding: 24,
        borderRadius: 12,
        maxWidth: 600,
        maxHeight: "80vh",
        overflowY: "auto",
      }}
    >
      <h3>{openStrategyModal === "momentum" ? "Momentum Strategy" : "Constant Volatility"}</h3>

      {openStrategyModal === "momentum" ? (
        <div style={{ textAlign: "left" }}>
          <p>
            The <strong>Momentum Strategy</strong> exploits trends by taking long positions in assets with positive recent returns and short positions in assets with negative recent returns.
          </p>

          <BlockMath math={String.raw`R_{i,t} = \frac{P_{i,t} - P_{i,t-1}}{P_{i,t-1}}`} />
          

          <BlockMath math={String.raw`\text{MA}_{i,s,t} = \frac{1}{N_s} \sum_{k=0}^{N_s-1} P_{i,t-k}, \quad
\text{MA}_{i,l,t} = \frac{1}{N_l} \sum_{k=0}^{N_l-1} P_{i,t-k}`} />

          <p>The momentum signal is:</p>
          <BlockMath math={String.raw`\text{Signal}_{i,t} = \text{MA}_{i,s,t} - \text{MA}_{i,l,t}`} />

          <p>Trading rule (daily rebalancing with cash):</p>
          <BlockMath math={String.raw`\text{Position}_{i,t}^{\text{signal}} = 
\begin{cases}
+1 & \text{if Signal}_{i,t} > 0 \\
-1 & \text{if Signal}_{i,t} < 0 \\
0 & \text{if Signal}_{i,t} = 0
\end{cases}`} />

          <p>At each day:</p>
          <ul>
            <li>Sell assets flagged for selling; proceeds become <strong>cash</strong>.</li>
            <li>Buy assets flagged for buying using available cash.</li>
            <li>If no assets to buy, hold the cash until a purchase is possible.</li>
            <li>Portfolio value: <BlockMath math={String.raw`V_t = \sum_i H_{i,t} \cdot P_{i,t} + \text{Cash}_t`} /></li>
          </ul>

          <p>Normalized weights for multiple buys:</p>
          <BlockMath math={String.raw`w_{i,t} = \frac{\max(\text{Signal}_{i,t}, 0)}{\sum_j \max(\text{Signal}_{j,t}, 0)}, \quad 
C_{i,t} = w_{i,t} \cdot \text{Cash}_t, \quad
H_{i,t}^{\text{new}} = \frac{C_{i,t}}{P_{i,t}}`} />

          <p>Portfolio return at day t:</p>
          <BlockMath math={String.raw`R^{\text{portfolio}}_t = \frac{V_t - V_{t-1}}{V_{t-1}}`} />

          <p>This strategy dynamically rebalances, only invests cash when there are assets to buy, and accounts for both held positions and cash.</p>
        </div>
) : (
  <div style={{ textAlign: "left" }}>
    <p>The <strong>Constant Volatility Strategy</strong> aims to maintain the portfolio's overall volatility at a target level <BlockMath math={`\\sigma_{\\text{target}}`} />. This is achieved by scaling positions according to historical volatility.</p>

    <p>Portfolio volatility over lookback window L:</p>
    <BlockMath math={`\\sigma_t = \\sqrt{\\frac{1}{L-1} \\sum_{k=0}^{L-1} (R^{\\text{portfolio}}_{t-k} - \\bar{R}^{\\text{portfolio}}_t)^2}`} />

    <p>Scaling factor to achieve target volatility:</p>
    <BlockMath math={`f_t = \\frac{\\sigma_{\\text{target}}}{\\sigma_t}`} />

    <p>Adjusted holdings for each asset:</p>
    <BlockMath math={`H_{i,t}^{\\text{adjusted}} = f_t \\cdot H_{i,t}^{\\text{current}}`} />

    <p>Portfolio return at day t:</p>
    <BlockMath math={`R^{\\text{portfolio}}_t = \\frac{V_t - V_{t-1}}{V_{t-1}}`} />

    <p>This strategy dynamically adjusts positions to maintain constant volatility while keeping the portfolio fully invested.</p>
  </div>
)}

      <button
        onClick={() => setOpenStrategyModal(null)}
        style={{
          marginTop: 16,
          padding: "6px 12px",
          borderRadius: 6,
          border: "none",
          backgroundColor: "#2b6cb0",
          color: "white",
          cursor: "pointer",
        }}
      >
        Close
      </button>
    </div>
  </div>
)}


        <button
          onClick={fetchStrategies}
          style={{
            padding: "10px 20px",
            backgroundColor: "#2b6cb0",
            color: "white",
            border: "none",
            borderRadius: 8,
            cursor: "pointer",
          }}
        >
          {loadingBacktest ? "Running..." : "Run Backtest"}
        </button>

        {strategies?.nav && (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={strategies.nav} style={{ marginTop: 24 }}>
              <CartesianGrid stroke="#ccc" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="linear" dataKey="Buy&Hold" stroke="#8884d8" dot={false} strokeWidth={2} />
              <Line type="linear" dataKey="Momentum" stroke="#82ca9d" dot={false} strokeWidth={2} />
              <Line type="linear" dataKey="ConstVol" stroke="#ffc658" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </motion.div>
    </div>
  );
}

export default PortfolioDashboard;
