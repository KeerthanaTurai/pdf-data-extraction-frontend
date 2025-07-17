import React from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";

const sidebarStyle = {
  position: "fixed",
  left: 0,
  top: 0,
  width: "200px",
  height: "100vh",
  background: "#f7f7f7",
  padding: 20,
  borderRight: "1px solid #ddd",
  boxSizing: "border-box", 
  zIndex: 10, // stays on top if scrolling
};

const linkStyle = (active) => ({
  display: "block",
  margin: "18px 0",
  color: active ? "#1976d2" : "#222",
  fontWeight: active ? "bold" : "normal",
  textDecoration: "none",
});

function Sidebar() {
  const location = useLocation();
  const navigate = useNavigate();
  const handleLogout = () => {
    localStorage.removeItem("token");
    navigate("/login");
  };
  return (
    <div style={sidebarStyle}>
      <div style={{ fontWeight: "bold", fontSize: 18, marginBottom: 30 }}>
        PDF Data Tool
      </div>
      <Link to="/" style={linkStyle(location.pathname === "/")}>Home</Link>
      <Link to="/upload" style={linkStyle(location.pathname === "/upload")}>Upload PDF</Link>
      <Link to="/view-and-approve" style={linkStyle(location.pathname === "/view-and-approve")}>View and Approve</Link>
      <Link to="/approved" style={linkStyle(location.pathname === "/approved")}>Approved</Link>
      <button onClick={handleLogout} style={{ marginTop: 30, color: "#e53935", background: "none", border: "none", cursor: "pointer" }}>Logout</button>
    </div>
  );
}

export default Sidebar;