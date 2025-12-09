import React from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
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
          <h2>Portfolio Dashboard</h2>
          <div style={{ display: "flex", gap: 16 }}>
            <Link to="/" style={{ color: "white", textDecoration: "none" }}>Portfolio</Link>
            <Link to="/strategies" style={{ color: "white", textDecoration: "none" }}>Strategies</Link>
          </div>
        </nav>

        {/* ---- Page Content ---- */}
        <div style={{ padding: 24 }}>
          <Routes>
            <Route path="/" element={<PortfolioPage />} />
            <Route path="/strategies" element={<StrategiesPage />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
