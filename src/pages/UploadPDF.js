import React, { useState } from "react";
import { Box, Paper, Typography, TextField, Button, MenuItem, LinearProgress, Alert } from "@mui/material";

function UploadPDF() {
  const [docType, setDocType] = useState("");
  const [date, setDate] = useState("");
  const [formularyName, setFormularyName] = useState("");
  const [policyName, setPolicyName] = useState("");
  const [file, setFile] = useState(null);
  const [progress, setProgress] = useState(0);
  const [loading, setLoading] = useState(false);
  const [successMsg, setSuccessMsg] = useState("");
  const [errorMsg, setErrorMsg] = useState("");

  const handleUpload = async (e) => {
    e.preventDefault();
    setSuccessMsg("");
    setErrorMsg("");

    if (!file) {
      setErrorMsg("Please select a PDF file.");
      return;
    }

    setLoading(true);
    setProgress(10);

    // Simulate file upload progress
    setTimeout(() => setProgress(40), 500);
    setTimeout(() => setProgress(80), 1000);
    setTimeout(() => {
      setProgress(100);
      setSuccessMsg("PDF uploaded successfully! (Backend not yet connected)");
      setProgress(0);
      setLoading(false);
    }, 1600);
  };

  return (
    <Box>
      <Typography variant="h5" mb={2}>Upload PDF</Typography>
      {loading && <LinearProgress sx={{ mb: 2 }} />}

      <Paper sx={{ padding: 2, mb: 2, maxWidth: 600 }}>
        <form onSubmit={handleUpload}>
          <TextField
            select
            label="Document Type"
            value={docType}
            onChange={e => setDocType(e.target.value)}
            required
            sx={{ mr: 2, mb: 2, width: 220 }}
          >
            <MenuItem value="criteria">Criteria</MenuItem>
            <MenuItem value="formulary">Formulary</MenuItem>
          </TextField>
          <TextField
            label="Date"
            type="date"
            value={date}
            onChange={e => setDate(e.target.value)}
            InputLabelProps={{ shrink: true }}
            required
            sx={{ mr: 2, mb: 2, width: 220 }}
          />
          <TextField
            label="Formulary Name"
            value={formularyName}
            onChange={e => setFormularyName(e.target.value)}
            required
            sx={{ mr: 2, mb: 2, width: 220 }}
          />
          <TextField
            label="Policy Name"
            value={policyName}
            onChange={e => setPolicyName(e.target.value)}
            required
            sx={{ mb: 2, width: 220 }}
          />
          <Box mb={2} mt={2}>
            <input
              type="file"
              accept="application/pdf"
              onChange={e => setFile(e.target.files[0])}
              style={{ marginBottom: 10 }}
              required
            />
          </Box>
          <Button type="submit" variant="contained" color="primary" disabled={loading}>Upload</Button>
        </form>
        {successMsg && <Alert severity="success" sx={{ mt: 2 }}>{successMsg}</Alert>}
        {errorMsg && <Alert severity="error" sx={{ mt: 2 }}>{errorMsg}</Alert>}
      </Paper>

      {/* Preview and DataFrame placeholder */}
      <Paper sx={{ padding: 2, maxWidth: 600 }}>
        <Typography variant="h6">Preview/DataFrame</Typography>
        <div>DataFrame will be shown here after upload and backend connection.</div>
      </Paper>
    </Box>
  );
}

export default UploadPDF;