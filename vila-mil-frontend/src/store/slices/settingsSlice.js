import { createSlice } from '@reduxjs/toolkit'

const initialState = {
  apiBaseUrl: '/api',
}

const settingsSlice = createSlice({
  name: 'settings',
  initialState,
  reducers: {
    setApiBaseUrl: (state, action) => {
      state.apiBaseUrl = action.payload || '/api'
    },
  },
})

export const { setApiBaseUrl } = settingsSlice.actions
export default settingsSlice.reducer

