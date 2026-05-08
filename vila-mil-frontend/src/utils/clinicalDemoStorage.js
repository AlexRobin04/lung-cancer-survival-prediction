/** Clinical「演示」：与 Settings 开关同步，associate-features 时读取 */
export const CLINICAL_DEMO_FAKE_EXTRACTION_LS = 'vila_clinical_demo_fake_extraction'

export function readClinicalDemoFakeExtraction() {
  try {
    return localStorage.getItem(CLINICAL_DEMO_FAKE_EXTRACTION_LS) === '1'
  } catch {
    return false
  }
}

export function writeClinicalDemoFakeExtraction(on) {
  try {
    if (on) localStorage.setItem(CLINICAL_DEMO_FAKE_EXTRACTION_LS, '1')
    else localStorage.removeItem(CLINICAL_DEMO_FAKE_EXTRACTION_LS)
  } catch {
    /* ignore */
  }
}
