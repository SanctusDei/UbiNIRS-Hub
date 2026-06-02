#include "NIRScanner.h"
#include "dlpspec.h"

using namespace std;


NIRScanner::NIRScanner(uScanConfig *pConfig) {

    // Init config.
    if (pConfig != nullptr) {
        this->mConfig = *pConfig;
    } else {
        this->mConfig.scanCfg.scanConfigIndex = 8;
        this->mConfig.scanCfg.scan_type = HADAMARD_TYPE;
        this->mConfig.scanCfg.num_patterns = 228;
        this->mConfig.scanCfg.num_repeats = 3;
        this->mConfig.scanCfg.wavelength_start_nm = MIN_WAVELENGTH;
        this->mConfig.scanCfg.wavelength_end_nm = MAX_WAVELENGTH;
        this->mConfig.scanCfg.width_px = 7;
    }

    // Open USB.
    USB_Init();
    if (0 != USB_Open()) {
        std::cout << "ERROR: Failed to open USB." << std::endl;
    }

    // Wake device from hibernate/sleep BEFORE any queries.
    // Without this, PGA query and RefCal fetches will read uninitialized registers.
    // Use a retry loop because the device may need multiple wake attempts
    // after deep hibernation.
    const int MAX_WAKE_RETRIES = 3;
    int pga_val = -1;
    bool refcal_ok = false;

    for (int attempt = 0; attempt < MAX_WAKE_RETRIES; attempt++) {
        NNO_SetHibernate(false);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // Reset error status.
        this->resetErrorStatus();

        // Quick health probe: can we read device status?
        unsigned int dev_status;
        int status = NNO_ReadDeviceStatus(&dev_status);
        if (status != PASS) {
            std::cout << "WARNING: Device not responsive on wake attempt "
                      << (attempt + 1) << "/" << MAX_WAKE_RETRIES << std::endl;
            continue;
        }

        // Device is awake. Query PGA gain and validate range.
        pga_val = NNO_GetPGAGain();
        if (pga_val <= 0 || pga_val > 64) {
            std::cout << "WARNING: PGA gain returned " << pga_val
                      << " (suspicious), retrying..." << std::endl;
            continue;
        }

        // Fetch reference calibration data.
        if (PASS != mEvm.FetchRefCalData()) {
            std::cout << "WARNING: FetchRefCalData failed on attempt "
                      << (attempt + 1) << "/" << MAX_WAKE_RETRIES
                      << ", retrying..." << std::endl;
            continue;
        }
        if (PASS != mEvm.FetchRefCalMatrix()) {
            std::cout << "WARNING: FetchRefCalMatrix failed on attempt "
                      << (attempt + 1) << "/" << MAX_WAKE_RETRIES
                      << ", retrying..." << std::endl;
            continue;
        }

        // Final health check: verify the DLPC scan engine is actually
        // ready.  Tiva may respond to status/PGA/RefCal queries while
        // the DLPC150 is still offline after deep hibernation.
        {
            int scan_est = NNO_GetEstimatedScanTime();
            if (scan_est <= 0 || scan_est > 30000) {
                std::cout << "WARNING: DLPC scan estimate bogus ("
                          << scan_est << "ms) on attempt "
                          << (attempt + 1) << "/" << MAX_WAKE_RETRIES
                          << ", retrying wake..." << std::endl;
                continue;
            }
        }

        refcal_ok = true;
        break;
    }

    // Fallback: use safe defaults if all retries exhausted.
    if (pga_val <= 0 || pga_val > 64) {
        std::cout << "WARNING: All wake attempts exhausted. "
                  << "PGA set to default 1." << std::endl;
        pga_val = 1;
    }
    this->mPrevPGAGain = pga_val;
    std::cout << "PGA gain: " << pga_val << std::endl;

    if (!refcal_ok) {
        std::cout << "WARNING: RefCal fetch failed after all retries. "
                  << "Scan results may be degraded." << std::endl;
    }

    // Diagnostic: check for DLPC-specific errors that may have accumulated
    // during hibernation.  These flags explain WHY the DLPC might report
    // bogus estimated scan times later.
    {
        NNO_error_status_struct errStatus;
        if (NNO_ReadErrorStatus(&errStatus) == PASS) {
            if (errStatus.status & NNO_ERROR_SCAN) {
                std::cout << "WARNING: Device reports scan error status=0x"
                          << std::hex << errStatus.status << std::dec;
                if (errStatus.errorCodes.scan == NNO_ERROR_SCAN_DLPC150_BOOT_ERROR)
                    std::cout << " (DLPC150 boot error)";
                else if (errStatus.errorCodes.scan == NNO_ERROR_SCAN_DLPC150_INIT_ERROR)
                    std::cout << " (DLPC150 init error)";
                std::cout << " ！ DLPC may need recovery before next scan"
                          << std::endl;
            }
            if (errStatus.status & NNO_ERROR_HW) {
                std::cout << "WARNING: Device reports HW (DLPC150) error ！ "
                          << "DLPC may be in bad state" << std::endl;
            }
        }
    }

    // Get reference data pointer.
    this->pRefDataBlob = this->mEvm.GetRefCalDataBlob();

    // Apply configuration.
    this->configEVM();
}


NIRScanner::~NIRScanner() {
    USB_Close();
}


int NIRScanner::readVersion()
/*
 * This function reads and displays the GUI , TIVA FW and DLPC Flash versions on the Information Tab
 * @return  0 = PASS
 *         -1 = FAIL
 *
 */
{
    char versionStr[255];
    unsigned int tiva_sw_ver;
    unsigned int dlpc_sw_ver;
    unsigned int dlpc_fw_ver;
    unsigned int speclib_ver;
    unsigned int eeprom_cfg_ver;
    unsigned int eeprom_cal_ver;
    unsigned int eeprom_refcal_ver;
    unsigned int tiva_sw_ver_major;
    unsigned int tiva_sw_ver_minor;
    unsigned int tiva_sw_ver_build;
    unsigned int speclib_ver_major;
    unsigned int speclib_ver_minor;
    unsigned int speclib_ver_build;
    unsigned int dlpc_fw_ver_major;
    unsigned int dlpc_fw_ver_minor;
    unsigned int dlpc_fw_ver_build;
    int ret_val;


    if (USB_IsConnected()) {
        uint32 device_status;

        if ((ret_val = NNO_ReadDeviceStatus(&device_status)) == PASS) {
            NNO_error_status_struct error_status;

            if (NNO_ReadErrorStatus(&error_status) == PASS) {
                std::cout << "Error Status " << error_status.status << " Press to clear" << std::endl;
            } else {
                std::cout << "Unable to read\n Error Status" << std::endl;
            }
        } else if (ret_val == NNO_CMD_BUSY) //If nano is servicing a higher priority command interface
        {
            /* Make sure NNO_GetVersion also returns NNO_CMD_BUSY because some older Tiva FW do not
             * support NNO_ReadDeviceStatus() command and their response to that command cannot be trusted
             */
            if (NNO_GetVersion(&tiva_sw_ver, &dlpc_sw_ver, &dlpc_fw_ver, &speclib_ver, &eeprom_cal_ver,
                               &eeprom_refcal_ver, &eeprom_cfg_ver) == NNO_CMD_BUSY) {
                //Set Tiva Active Status as false and BT connected as true
                std::cout << "NNO_CMD_BUSY" << std::endl;
            }
        } else {
            //Set Tiva Active Status as false
            std::cout << "Active status false" << std::endl;
        }


        if (NNO_GetVersion(&tiva_sw_ver, &dlpc_sw_ver, &dlpc_fw_ver, &speclib_ver, &eeprom_cal_ver, &eeprom_refcal_ver,
                           &eeprom_cfg_ver) == 0) {
            // Tiva version.
            tiva_sw_ver &= 0xFFFFFF;
            tiva_sw_ver_major = tiva_sw_ver >> 16;
            tiva_sw_ver_minor = (tiva_sw_ver << 16) >> 24;
            tiva_sw_ver_build = (tiva_sw_ver << 24) >> 24;

            sprintf(versionStr, "Software version: %d.%d.%d", tiva_sw_ver_major, tiva_sw_ver_minor, tiva_sw_ver_build);
            std::cout << versionStr << std::endl;

            // DLPC version.
            dlpc_fw_ver_major = dlpc_fw_ver >> 24;
            dlpc_fw_ver_minor = (dlpc_fw_ver << 8) >> 24;
            dlpc_fw_ver_build = (dlpc_fw_ver << 16) >> 16;
            sprintf(versionStr, "Firmware version: %d.%d.%d", dlpc_fw_ver_major, dlpc_fw_ver_minor, dlpc_fw_ver_build);
            std::cout << versionStr << std::endl;

            // EEPROM cal version.
            sprintf(versionStr, "EEPROM version: %d/", eeprom_cal_ver);
            std::cout << versionStr;

            // EEPROM refcal version.
            sprintf(versionStr, "%d/", eeprom_refcal_ver);
            std::cout << versionStr;

            // EEPROM config version.
            sprintf(versionStr, "%d", eeprom_cfg_ver);
            std::cout << versionStr << std::endl;

            // Speclib verion.
            speclib_ver &= 0xFFFFFF;
            speclib_ver_major = speclib_ver >> 16;
            speclib_ver_minor = (speclib_ver << 16) >> 24;
            speclib_ver_build = (speclib_ver << 24) >> 24;

            sprintf(versionStr, "Spectrum lib version: %d.%d.%d", speclib_ver_major, speclib_ver_minor,
                    speclib_ver_build);
            std::cout << versionStr << std::endl;

            return PASS;
        } else {
            return FAIL;    //Read failed
        }
    } else //if USB is not connected
    {
        //Set Tiva Active Status as false
        std::cout << "USB not connected." << std::endl;

    }
    return FAIL; //USB is not connected.
}


void NIRScanner::resetErrorStatus()
/*
* Reset device's error status.
*/
{
    int result = NNO_ResetErrorStatus();
    if (result == FAIL) {
        printf("ERROR: Failed to reset error status.");
    }
    else {
        this->mErrorFlag = false;
    }
}


void NIRScanner::configEVM(uScanConfig *pConfig)
/* 
 * Config NIRScan Nano. If pConfig is not provided, using the config stored in the instance..
*/
{
    if (pConfig != nullptr) {
        this->mConfig = *pConfig;
    }

    int ret = this->mEvm.ApplyScanCfgtoDevice(&this->mConfig);
    if (ret < 0) {
        std::cout << "Apply scan config FAILED (ret=" << ret << ")" << std::endl;
        return;
    }

}

void NIRScanner::setConfig(uint16_t scanConfigIndex,  // < Unique ID per spectrometer which is modified when the config is changed. Can be used to determine whether a cached version of the config is valid per spectrometer SN.
                           uint8_t scan_type,  // 0: COLUMN_TYPE, 1: HADAMARD_TYPE, 2: SLEW_TYPE.
                           uint16_t num_patterns, // Number of desired points in the spectrum.
                           uint16_t num_repeats, // Number of times to repeat the scan on the spectromter before averaging the scans together and returning the results. This can be used to increase integration time.
                           uint16_t wavelength_start_nm, // Minimum wavelength to start the scan from, in nm.
                           uint16_t wavelength_end_nm, // Maximum wavelength to end the scan at, in nm.
                           uint8_t width_px // Pixel width of the patterns. Increasing this will increase SNR, but reduce resolution.
                           ) {
    this->mConfig.scanCfg.scanConfigIndex = scanConfigIndex;
    this->mConfig.scanCfg.scan_type = scan_type;
    this->mConfig.scanCfg.num_patterns = num_patterns;
    this->mConfig.scanCfg.num_repeats = num_repeats;
    this->mConfig.scanCfg.wavelength_start_nm = wavelength_start_nm;
    this->mConfig.scanCfg.wavelength_end_nm = wavelength_end_nm;
    this->mConfig.scanCfg.width_px = width_px;

    this->configEVM();
}
void NIRScanner::setPGAGain(int32_t newValue)
/*
* Set the PGA gain.
* @param newValue - I - Sets ADC PGA gain. Valid values are 0,1,2,4,8,16,32 or 64. If 0, then the gain is auto. 
*
*/
{
    int result;
    
    // ==========================================
    // [ ?????? 1 ]
    // ????????? NNO_GetPGAGain() ??
    // ??????λ??(Python)??????????????????????????????
    // ==========================================
    this->mPrevPGAGain = newValue;

    if(newValue == 0)
    {
        result = NNO_SetFixedPGAGain(false, 1);
    }
    else{
        result = NNO_SetFixedPGAGain(true, newValue);
    }

    if(result == FAIL) {
        printf("ERROR: setPGAGain: failed to set PGA gain, new value %d.\n", newValue);
    }
}

void NIRScanner::setLampOnOff(int32_t newValue)
/* * Set the lamp always on or off. 
* @param newValue - I - if -1 then always off, if 0 then on when scanning, if 1 then always on. 
*/
{
    // Sanity check.
    if(newValue > 1 || newValue < -1) {
        // Do nothing. 
        printf("ERROR: setLampOnOff: unsupported value %d\n", newValue);
        return;
    }

    // On when scanning.
    if(newValue == 0) { 
        // Enable control when scanning.
        NNO_SetScanControlsDLPCOnOff(true);
        // Disable DLPC.
        NNO_DLPCEnable(false, false);
        // Set PGA Gain to auto.
        NNO_SetFixedPGAGain(false, this->mPrevPGAGain);
    }
    else { 
        // Reset then set (might have low-level bugs there.)

        // Enable auto-contol.
        NNO_SetScanControlsDLPCOnOff(true);
        // Disable DLPC.
        NNO_DLPCEnable(false, false);
        // Wait for a second for the command execution.
        sleep(1);

        // Set manually control the lamp.
        // ???????????????? ADC ???????PGA ???????? 1
        NNO_SetScanControlsDLPCOnOff(false);

        if(newValue == -1) {
            // Enable DLPC, keep the lamp off.
            printf("INFO: keeping lamp off.\n");
            NNO_DLPCEnable(true, false);
        }
        else if(newValue == 1) {
            // Enable DLPC, keep the lamp on.
            printf("INFO: keeping lamp on.\n");
            NNO_DLPCEnable(true, true);
        }

        // ==========================================
        // [ ?????? 2 ] 
        // ???????????????????? NNO_GetPGAGain()
        // ????????? mPrevPGAGain ????????????ι???????
        // ==========================================
        if(this->mPrevPGAGain > 0) {
            NNO_SetFixedPGAGain(true, this->mPrevPGAGain);
        } else {
            NNO_SetFixedPGAGain(true, 1);
        }
    }
}
int NIRScanner::_performScanReadData(bool storeInSD, uint16 numRepeats, void *pData, int *pBytesRead)
/*
 * This function asks the Nano to perform the scan and gets back the ScanData
 * @param storeInSD - I - a boolean to indicate if the current scan is to be stored in SD card
 * @param numRepeats - I - an integer indicating the number of times the scan should repeat in Nano
 * @param pData - O - The scanData is readback from Nano in this variable
 * @param pBytesRead - O - the size of the scnData read
 *
 */
{
    int size;
    int scanTimeOut;
    time_t timeScanEnd;
    time_t timeScanStart;
    time_t lastScanTimeMS;
    unsigned int devStatus;
    string scanTimeText;

    // Reset error.
    NNO_ResetErrorStatus();

    // NNO_SetFixedPGAGain(true,1);   /* Only used for testing Fixed PGA command  */
    NNO_SetScanNumRepeats(numRepeats);

    int estimated = NNO_GetEstimatedScanTime();

    // Fail fast if the scan subsystem isn't ready.  DLPC validation is
    // now done in the constructor (wake-up loop), so a bogus estimate
    // here means the device entered a bad state between init and scan.
    // We do NOT attempt DLPC toggling here ！ that has been shown to
    // corrupt USB communication on a fragile post-hibernate device.
    if (estimated <= 0 || estimated > 30000) {
        std::cout << "ERROR: Bogus estimated scan time " << estimated
                  << "ms ！ scan subsystem not ready" << std::endl;
        return FAIL;
    }

    std::cout << "Scan in progress. Estimated Scan time is approximately "
              << (float) estimated / 1000.0 << " seconds. ";

    scanTimeOut = estimated * 3;
    if (scanTimeOut > 30) scanTimeOut = 30;  // hard cap at 30 seconds
    timeScanStart = time(0);

    // Retry loop: if the device returns garbage data (can happen after
    // hibernate wake), reset errors and try once more.
    const int MAX_SCAN_ATTEMPTS = 2;
    for (int attempt = 0; attempt < MAX_SCAN_ATTEMPTS; attempt++) {
        if (attempt > 0) {
            std::cout << "Retrying scan (attempt " << (attempt + 1) << ")... ";
            NNO_ResetErrorStatus();
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            timeScanStart = time(0);
        }

        NNO_PerformScan(storeInSD);
        //Wait for scan completion
        if (NNO_ReadDeviceStatus(&devStatus) == PASS) {
            do {
                if ((devStatus & NNO_STATUS_SCAN_IN_PROGRESS) != NNO_STATUS_SCAN_IN_PROGRESS) {
                    break;
                }
                timeScanEnd = time(0);
                if ((timeScanEnd - timeScanStart) >= scanTimeOut) {
                    *pBytesRead = 0;
                    std::cout << "Scan time out with " << timeScanEnd - timeScanStart << std::endl;
                    return FAIL;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));

            } while (NNO_ReadDeviceStatus(&devStatus) == PASS);
        } else {
            std::cout << "Reading device status for scan completion failed." << std::endl;
            if (attempt < MAX_SCAN_ATTEMPTS - 1) continue;
            return FAIL;
        }

        timeScanEnd = time(0);
        lastScanTimeMS = timeScanEnd - timeScanStart;
        std::cout << "Scan time was " << lastScanTimeMS << "ms" << std::endl;

        *pBytesRead = NNO_GetFileSizeToRead(NNO_FILE_SCAN_DATA);

        if (*pBytesRead <= 0 || *pBytesRead > SCAN_DATA_BLOB_SIZE) {
            std::cout << "WARNING: Bogus scan data size " << *pBytesRead
                      << " (expected ~" << SCAN_DATA_BLOB_SIZE << ")"
                      << std::endl;
            if (attempt < MAX_SCAN_ATTEMPTS - 1) continue;
            *pBytesRead = 0;
            return FAIL;
        }

        if ((size = NNO_GetFile((unsigned char *) pData, *pBytesRead)) != *pBytesRead) {
            *pBytesRead = size;
            std::cout << "Scan data read from device failed" << std::endl;
            if (attempt < MAX_SCAN_ATTEMPTS - 1) continue;
            return FAIL;
        }

        // Post-scan diagnostics: verify the device didn't report scan errors.
        // NNO_PerformScan() does not clear previous scan data, so a silent
        // scan failure would cause NNO_GetFile() to return stale data.
        {
            NNO_error_status_struct errStatus;
            if (NNO_ReadErrorStatus(&errStatus) == PASS
                && (errStatus.status & NNO_ERROR_SCAN)) {
                std::cout << "WARNING: Device reports scan error status=0x"
                          << std::hex << errStatus.status << std::dec
                          << " ！ data may be stale";
                if (errStatus.errorCodes.scan == NNO_ERROR_SCAN_DLPC150_BOOT_ERROR)
                    std::cout << " (DLPC150 boot error)";
                else if (errStatus.errorCodes.scan == NNO_ERROR_SCAN_DLPC150_INIT_ERROR)
                    std::cout << " (DLPC150 init error)";
                std::cout << std::endl;
                if (attempt < MAX_SCAN_ATTEMPTS - 1) continue;
                return FAIL;
            }
        }

        return PASS;
    }

    return FAIL;
}


int NIRScanner::_interpretData(void *pData)
/**
 * This function takes scan data and reference data as serialzed blobs, interprets them
 * using spectrum library APIs and saves for future use.
 *
 * @param pScanDataBlob - I - pointer to scanData blob; if NULL it will continue to use the last set value
 * @param pRefDataBlob - I - pointer to reference data blob; if NULL it will continue to use the last set value
 */
{
    uScanData *_pRefData;
    DLPSPEC_ERR_CODE ret_val;
    void *pCopyBuff = nullptr;

    if (pData != nullptr) {
        if (dlpspec_scan_interpret(pData, SCAN_DATA_BLOB_SIZE, &this->mScanResults) != PASS)
            return FAIL;
    }

    if (this->pRefDataBlob != nullptr) {
        pCopyBuff = malloc(SCAN_DATA_BLOB_SIZE);

        if (pCopyBuff == nullptr) {
            return (ERR_DLPSPEC_INSUFFICIENT_MEM);
        }

        memcpy(pCopyBuff, this->pRefDataBlob, SCAN_DATA_BLOB_SIZE);

        // Deserialize
        ret_val = dlpspec_scan_read_data(pCopyBuff, SCAN_DATA_BLOB_SIZE);
        if (ret_val < 0) {
            free(pCopyBuff);
            pCopyBuff = nullptr;
            return ret_val;
        }

        _pRefData = (uScanData *) pCopyBuff;
        ret_val = dlpspec_scan_interpReference(this->pRefDataBlob, SCAN_DATA_BLOB_SIZE,
                                               this->mEvm.GetRefCalMatrixBlob(_pRefData->data.serial_number),
                                               REF_CAL_MATRIX_BLOB_SIZE, &this->mScanResults,
                                               &this->mReferenceResults);
        if (PASS != ret_val) {
            return FAIL;
        }
        if (pCopyBuff != nullptr) {
            free(pCopyBuff);
            pCopyBuff = nullptr;
        }
    }
    return PASS;
}


string NIRScanner::scanSNR(bool isHadamard) 
/**
 * Perform a special scan sequence and return SNR values in formated string: [val1, val2, val3].
 * val1, val2 can val3 are SNR ratios at different time intervals namely 17ms, 133ms and 600ms. 
 * 
 * @param isHadamard - B - if true then perform Hadamard scan, otherwise Column.
 * 
 */   
{
    int status;
    if (isHadamard == true) {
        status = NNO_StartHadSNRScan();
    }
    else {
        status = NNO_StartSNRScan();
    }
    NNO_PerformScan(false);

    // Scan failed.
    if (status != PASS) {
        std::cout << "ERROR: SNR scan failed." << std::endl;
        return "";
    }

    // Get SNR data.
    int val1, val2, val3;
    status = NNO_GetSNRData(&val1, &val2, &val3);

    // Get results failed.
    if (status != PASS) {
        std::cout << "ERROR: Get SNR data failed." << std::endl;
        return "";
    }

    // Format and return SNR data.
    std::stringstream resultSNRs;
    resultSNRs << "[" << val1 << "," << val2 << "," << val3 << "]";
    return resultSNRs.str();
}


void NIRScanner::scan(bool saveDataFlag, int numRepeats)
/**
 * This is a handler function for pushButton_scan on Scan Tab clicked() event
 * This function does the following tasks
 * Checks for USB connection
 * gets the selected Scan Configuration parameters - estimates the scan time and displays
 * does the scan by calling the corresponding API functions
 * saves the scan results in .csv and .bat files in user settings directory
 * displays the spectrum - plots the scan data on the GUI
 */
{
    void *pData;
    int scanStatus;
    int fileSize;
    string scanTimeText;

    pData = (scanData *) malloc(SCAN_DATA_BLOB_SIZE);
    if (pData == nullptr) {
        std::cout << "ERROR: Out of memory" << std::endl;
        return;
    }
    // Zero the buffer so that if the device returns partial or stale data,
    // uninitialized bytes don't confuse the TPL deserializer.
    memset(pData, 0, SCAN_DATA_BLOB_SIZE);

    // Scanning.
    scanStatus = this->_performScanReadData(NNO_DONT_STORE_SCAN_IN_SD, numRepeats, pData, &fileSize);
    if (scanStatus != PASS) {
        std::cout << "ERROR: Scan failed." << std::endl;
    }

    // Display versions.
    std::cout << "Header version: " << ((scanData *) pData)->header_version << std::endl;

    // Zero out previous results so that if interpretation fails,
    // getScanData() returns empty data (detectable by Python) instead of
    // leaking stale results from a prior successful scan.
    this->mScanResults = scanResults{};

    int retVal = this->_interpretData(pData);
    if (retVal != PASS) {
        std::cout << "ERROR: Interpret data failed." << std::endl;
    } else {
        std::cout << std::endl;
        std::cout << "Detector temperature: " << this->mScanResults.detector_temp_hundredths << std::endl;

        // Save data.
        if (saveDataFlag) {
            // string fileName;
            char writeBuffer[1024];
            // std::cout << "File name to save: ";
            // std::cin >> fileName;

            ofstream myfile;
            myfile.open("scandata.csv");
            if (myfile) {
                for (int i = 0; i < mConfig.scanCfg.num_patterns; ++i) {
                    sprintf(writeBuffer, "%6.2f,%5d,%.2f\n",
                            this->mScanResults.wavelength[i],
                            this->mScanResults.intensity[i],
                            (double) this->mScanResults.intensity[i] / (double) this->mReferenceResults.intensity[i]);
                    myfile << writeBuffer;
                }
                myfile.close();
            }
        }
    }
    free(pData);
}


string NIRScanner::getScanData()
/**
* Convert scanning results to string dictionary.
* This is for Python API.
*/
{
    auto _arrayToString = [](const void *const pArray, int length, char type) -> string {
        string result;

        switch (type) {
            case 'f': {
                for (int i = 0; i < length; i++) {
                    result += (to_string(((double *) pArray)[i]) + string(","));
                }
                break;
            }
            case 'i': {
                for (int i = 0; i < length; i++) {
                    result += (to_string(((int *) pArray)[i]) + string(","));
                }
                break;
            }
            default: {
                break;
            }
        }

        return result;
    };

    string scanResults;
    scanResults = string("header_version:") + to_string(this->mScanResults.header_version);
    scanResults += string("\nscan_name:") + string(this->mScanResults.scan_name);
    scanResults += string("\nscan_time:")
                   + to_string(this->mScanResults.year + 2000)
                   + to_string(this->mScanResults.month + 1)
                   + to_string(this->mScanResults.day)
                   + to_string(this->mScanResults.hour)
                   + to_string(this->mScanResults.minute)
                   + to_string(this->mScanResults.second);
    scanResults += string("\ntemperature_system:") + to_string(this->mScanResults.system_temp_hundredths);
    scanResults += string("\ntemperature_detector:") + to_string(this->mScanResults.detector_temp_hundredths);
    scanResults += string("\nhumidity:") + to_string(this->mScanResults.humidity_hundredths);
    scanResults += string("\npga:") + to_string(this->mScanResults.pga);
    scanResults += string("\nwavelength:") + _arrayToString(this->mScanResults.wavelength, this->mScanResults.length, 'f');
    scanResults += string("\nintensity:") + _arrayToString(this->mScanResults.intensity, this->mScanResults.length, 'i');
    scanResults += string("\nreference:") + _arrayToString(this->mReferenceResults.intensity, this->mReferenceResults.length, 'i');
    scanResults += string("\nvalid_length:") + to_string(this->mScanResults.length);
//    scanResults["header_version"] = to_string(this->mScanResults.header_version);
//    scanResults["scan_name"] = string(this->mScanResults.scan_name);
//    scanResults["scan_time"] = to_string(this->mScanResults.year+2000)
//                               + to_string(this->mScanResults.month+1)
//                               + to_string(this->mScanResults.day)
//                               + to_string(this->mScanResults.hour)
//                               + to_string(this->mScanResults.minute)
//                               + to_string(this->mScanResults.second);
//    scanResults["temperature_system"] = to_string(this->mScanResults.system_temp_hundredths);
//    scanResults["temperature_detector"] = to_string(this->mScanResults.detector_temp_hundredths);
//    scanResults["humidity"] = to_string(this->mScanResults.humidity_hundredths);
//    scanResults["pga"] = to_string(this->mScanResults.pga);
//    scanResults["wavelength"] = _arrayToString(this->mScanResults.wavelength, this->mScanResults.length, 'f');
//    scanResults["intensity"] = _arrayToString(this->mScanResults.intensity, this->mScanResults.length, 'i');
//    scanResults["valid_length"] = to_string(this->mScanResults.length);

    return scanResults;
}

int NIRScanner::setHibernate(bool newValue)
/**
* Enable hibernate after inactive if True, otherwise disable. 
* This is for Python API.
*/
{
    return NNO_SetHibernate(newValue);
}



