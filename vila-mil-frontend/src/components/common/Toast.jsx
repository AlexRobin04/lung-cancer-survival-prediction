import React from 'react'
import { Alert, Snackbar } from '@mui/material'

export default function Toast({
  open,
  message,
  severity = 'success',
  onClose,
  autoHideDuration = 2500,
  anchorOrigin = { vertical: 'top', horizontal: 'right' },
}) {
  return (
    <Snackbar
      open={open}
      onClose={onClose}
      autoHideDuration={autoHideDuration}
      anchorOrigin={anchorOrigin}
    >
      <Alert onClose={onClose} severity={severity} variant="filled" sx={{ width: '100%' }}>
        {message}
      </Alert>
    </Snackbar>
  )
}

