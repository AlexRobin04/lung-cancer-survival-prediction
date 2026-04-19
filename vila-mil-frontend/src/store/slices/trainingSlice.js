import { createSlice } from '@reduxjs/toolkit'

const initialState = {
  trainingTasks: [],
  currentTask: null,
  loading: false,
  error: null
}

const trainingSlice = createSlice({
  name: 'training',
  initialState,
  reducers: {
    startTraining: (state, action) => {
      state.loading = true
      state.error = null
    },
    startTrainingSuccess: (state, action) => {
      state.loading = false
      state.currentTask = action.payload
      state.trainingTasks.push(action.payload)
    },
    startTrainingFailure: (state, action) => {
      state.loading = false
      state.error = action.payload
    },
    updateTrainingStatus: (state, action) => {
      const { taskId, status, progress, loss, cIndex } = action.payload
      if (state.currentTask && state.currentTask.id === taskId) {
        state.currentTask = {
          ...state.currentTask,
          status,
          progress,
          loss,
          cIndex
        }
      }
      const taskIndex = state.trainingTasks.findIndex(task => task.id === taskId)
      if (taskIndex !== -1) {
        state.trainingTasks[taskIndex] = {
          ...state.trainingTasks[taskIndex],
          status,
          progress,
          loss,
          cIndex
        }
      }
    },
    stopTraining: (state, action) => {
      state.loading = true
      state.error = null
    },
    stopTrainingSuccess: (state, action) => {
      state.loading = false
      const { taskId } = action.payload
      if (state.currentTask && state.currentTask.id === taskId) {
        state.currentTask = {
          ...state.currentTask,
          status: 'stopped'
        }
      }
      const taskIndex = state.trainingTasks.findIndex(task => task.id === taskId)
      if (taskIndex !== -1) {
        state.trainingTasks[taskIndex] = {
          ...state.trainingTasks[taskIndex],
          status: 'stopped'
        }
      }
    },
    stopTrainingFailure: (state, action) => {
      state.loading = false
      state.error = action.payload
    },
    clearCurrentTask: (state) => {
      state.currentTask = null
    }
  }
})

export const { 
  startTraining, 
  startTrainingSuccess, 
  startTrainingFailure, 
  updateTrainingStatus, 
  stopTraining, 
  stopTrainingSuccess, 
  stopTrainingFailure, 
  clearCurrentTask 
} = trainingSlice.actions

export default trainingSlice.reducer
