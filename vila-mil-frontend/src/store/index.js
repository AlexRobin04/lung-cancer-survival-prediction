import { configureStore } from '@reduxjs/toolkit'
import trainingReducer from './slices/trainingSlice'
import dataReducer from './slices/dataSlice'
import modelReducer from './slices/modelSlice'
import settingsReducer from './slices/settingsSlice'

const store = configureStore({
  reducer: {
    training: trainingReducer,
    data: dataReducer,
    model: modelReducer,
    settings: settingsReducer
  }
})

export default store
