import React from "react";
import { BrowserRouter as Router, Routes, Route, Link, Navigate } from "react-router-dom";
import PortfolioPage from "./components/PortfolioPage";
import StrategiesPage from "./components/StrategiesPage";

function App() {
  return (
    <Router>
      <div style={{ fontFamily: "Segoe UI, sans-serif" }}>
        {/* ---- Navbar ---- */}
        <nav style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "12px 24px",
          backgroundColor: "#2b6cb0",
          color: "white"
        }}>
          <h2>Financial Dashboard Training</h2>
          <div style={{ display: "flex", gap: 16 }}>
          <div style={{ display: "flex", gap: 16 }}>
  <Link
    to="/portfolio"
    style={{
      padding: "8px 16px",
      backgroundColor: "#48bb78",
      color: "white",
      textDecoration: "none",
      borderRadius: 8,
      fontWeight: 600,
      transition: "all 0.2s",
    }}
    onMouseEnter={(e) => e.currentTarget.style.backgroundColor = "#38a169"}
    onMouseLeave={(e) => e.currentTarget.style.backgroundColor = "#48bb78"}
  >
    Portfolio Management
  </Link>

  <Link
    to="/strategies"
    style={{
      padding: "8px 16px",
      backgroundColor: "#48bb78",
      color: "white",
      textDecoration: "none",
      borderRadius: 8,
      fontWeight: 600,
      transition: "all 0.2s",
    }}
    onMouseEnter={(e) => e.currentTarget.style.backgroundColor = "#38a169"}
    onMouseLeave={(e) => e.currentTarget.style.backgroundColor = "#48bb78"}
  >
    Backtest Strategies
  </Link>
</div>

          </div>
        </nav>

        {/* ---- Page Content ---- */}
        <div style={{ padding: 24 }}>
          <Routes>
            {/* Redirige seulement la racine "/" vers "/strategies" */}
            <Route path="/" element={<Navigate to="/strategies" replace />} />
            <Route path="/strategies" element={<StrategiesPage />} />
            <Route path="/portfolio" element={<PortfolioPage />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
