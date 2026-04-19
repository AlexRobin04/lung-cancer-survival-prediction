import { createTheme } from '@mui/material/styles'

/** 下拉面板最大高度，超出后出现滚动条并支持滚轮浏览 */
const SELECT_MENU_MAX_HEIGHT_PX = 320

/**
 * 全局主题：统一 Select / Menu 下拉区域可滚动，避免长列表撑满整页且难以用滚轮浏览。
 */
export const appTheme = createTheme({
  components: {
    MuiSelect: {
      defaultProps: {
        MenuProps: {
          PaperProps: {
            sx: {
              maxHeight: SELECT_MENU_MAX_HEIGHT_PX,
              overflowY: 'auto',
            },
          },
        },
      },
    },
    // 未走 Select defaultProps 的 Menu（如部分自定义入口）仍限制纸张高度
    MuiMenu: {
      styleOverrides: {
        paper: {
          maxHeight: SELECT_MENU_MAX_HEIGHT_PX,
          overflowY: 'auto',
        },
      },
    },
  },
})
