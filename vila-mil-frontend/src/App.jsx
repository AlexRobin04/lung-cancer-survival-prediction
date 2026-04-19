import React from 'react'
import { Routes, Route, Navigate, Link, useLocation } from 'react-router-dom'
import { AppBar, Box, Container, Toolbar, Typography, Button, Stack } from '@mui/material'

import Dashboard from './components/Dashboard/Dashboard.jsx'
import DataManagement from './components/DataManagement/DataManagement.jsx'
import Training from './components/Training/Training.jsx'
import ModelEvaluation from './components/ModelEvaluation/ModelEvaluation.jsx'
import Clinical from './components/Clinical/Clinical.jsx'
import Prediction from './components/Prediction/Prediction.jsx'
import Settings from './components/Settings/Settings.jsx'

const TopNav = () => {
  const loc = useLocation()
  const items = [
    { to: '/', label: 'Dashboard' },
    { to: '/data-management', label: 'Data' },
    { to: '/training', label: 'Training' },
    { to: '/model-evaluation', label: 'Evaluation' },
    { to: '/clinical', label: 'Clinical' },
    { to: '/prediction', label: 'Prediction' },
    { to: '/settings', label: 'Settings' },
  ]
  return (
    <AppBar position="sticky" elevation={1}>
      <Toolbar sx={{ minHeight: { xs: 56, sm: 64 }, gap: 1 }}>
        <Typography
          variant="h6"
          noWrap
          sx={{
            flexGrow: 1,
            minWidth: 0,
            mr: 1,
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            wordBreak: 'keep-all',
            overflowWrap: 'normal',
            display: { xs: 'none', md: 'block' },
          }}
        >
          基于病理图像的肺癌生存风险预测系统
        </Typography>
        <Stack
          direction="row"
          spacing={1}
          sx={{
            flexWrap: 'nowrap',
            overflowX: 'auto',
            maxWidth: { xs: '100%', md: 'auto' },
            '&::-webkit-scrollbar': { display: 'none' },
            scrollbarWidth: 'none',
          }}
        >
          {items.map((it) => (
            <Button
              key={it.to}
              component={Link}
              to={it.to}
              color="inherit"
              variant={loc.pathname === it.to ? 'outlined' : 'text'}
              sx={{
                borderColor: 'rgba(255,255,255,0.5)',
                whiteSpace: 'nowrap',
                flexShrink: 0,
                minWidth: { xs: 88, md: 'auto' },
              }}
            >
              {it.label}
            </Button>
          ))}
        </Stack>
      </Toolbar>
    </AppBar>
  )
}

export default function App() {
  return (
    <Box sx={{ minHeight: '100vh' }}>
      <TopNav />
      <Container className="container" sx={{ py: 2 }}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/data-management" element={<DataManagement />} />
          <Route path="/training" element={<Training />} />
          <Route path="/model-evaluation" element={<ModelEvaluation />} />
          <Route path="/clinical" element={<Clinical />} />
          <Route path="/prediction" element={<Prediction />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Container>
    </Box>
  )
}

