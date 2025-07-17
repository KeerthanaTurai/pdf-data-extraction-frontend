import React from "react";
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import UploadPDF from "./pages/UploadPDF";
import ViewAndApprove from "./pages/ViewAndApprove";
import Approved from "./pages/Approved";
import Login from "./pages/Login";

function RequireAuth({ children }) {
  const location = useLocation();
  const token = localStorage.getItem("token");
  if (!token) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }
  return children;
}

function App() {
  const token = localStorage.getItem("token");
  return (
    <Router>
      {token && <Sidebar />}
      <div style={{
        marginLeft: token ? 200 : 0,
        padding: 24,
        minHeight: "100vh",
        background: "#fff"
      }}>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/" element={
            <RequireAuth>
              <Dashboard />
            </RequireAuth>
          } />
          <Route path="/upload" element={
            <RequireAuth>
              <UploadPDF />
            </RequireAuth>
          } />
          <Route path="/view-and-approve" element={
            <RequireAuth>
              <ViewAndApprove />
            </RequireAuth>
          } />
          <Route path="/approved" element={
            <RequireAuth>
              <Approved />
            </RequireAuth>
          } />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;