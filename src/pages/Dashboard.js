import React from "react";
import { Box, Grid, Paper, Typography, LinearProgress, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from "@mui/material";

function StatBox({ label, value, color }) {
  return (
    <Paper elevation={3} sx={{ padding: 2, textAlign: "center", minHeight: 100, backgroundColor: color || "#fff" }}>
      <Typography variant="h6" fontWeight="bold">{label}</Typography>
      <Typography variant="h4" mt={1}>{value}</Typography>
    </Paper>
  );
}

// Placeholder data, replace with API call later
const operations = [
  { id: 1, docType: "Criteria", date: "2024-07-15", name: "Policy A", approved: 2, disapproved: 0 },
  { id: 2, docType: "Formulary", date: "2024-07-12", name: "Formulary X", approved: 1, disapproved: 1 },
  { id: 3, docType: "Criteria", date: "2024-07-11", name: "Policy B", approved: 0, disapproved: 1 }
];

function Dashboard() {
  // Summarize stats
  const totalOperations = operations.length;
  const approvedCount = operations.reduce((sum, op) => sum + op.approved, 0);
  const disapprovedCount = operations.reduce((sum, op) => sum + op.disapproved, 0);

  return (
    <Box>
      {/* Stats Row */}
      <Grid container spacing={2} mb={2}>
        <Grid item xs={12} sm={4}><StatBox label="Total Operations" value={totalOperations} color="#e3f2fd" /></Grid>
        <Grid item xs={12} sm={4}><StatBox label="Approved Data Count" value={approvedCount} color="#e8f5e9" /></Grid>
        <Grid item xs={12} sm={4}><StatBox label="Disapproved" value={disapprovedCount} color="#ffebee" /></Grid>
      </Grid>

      {/* Progress Bar */}
      <Box mb={2}>
        <Typography mb={1}>Progress</Typography>
        <LinearProgress
          variant="determinate"
          value={totalOperations ? (approvedCount / totalOperations) * 100 : 0}
          sx={{ height: 10, borderRadius: 5 }}
        />
      </Box>

      {/* Table */}
      <Paper sx={{ padding: 2 }}>
        <Typography variant="h6" mb={2}>Recent Operations</Typography>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>S.No</TableCell>
                <TableCell>Document Type</TableCell>
                <TableCell>Date</TableCell>
                <TableCell>Formulary/Policy Name</TableCell>
                <TableCell>Approved Data</TableCell>
                <TableCell>Disapproved</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {operations.map((op, idx) => (
                <TableRow key={op.id}>
                  <TableCell>{idx + 1}</TableCell>
                  <TableCell>{op.docType}</TableCell>
                  <TableCell>{op.date}</TableCell>
                  <TableCell>{op.name}</TableCell>
                  <TableCell>{op.approved}</TableCell>
                  <TableCell>{op.disapproved}</TableCell>
                </TableRow>
              ))}
              {operations.length === 0 && (
                <TableRow>
                  <TableCell colSpan={6} align="center">No data</TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>
    </Box>
  );
}

export default Dashboard;