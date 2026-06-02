# NIRScanner Python Wrapper
# Created by Weiwei Jiang
#

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "./"))

import atexit
from _NIRScanner import *

import ctypes


class NIRS:

    class TYPES:
        COLUMN_TYPE = 0
        HADAMARD_TYPE = 1
        SLEW_TYPE = 2

    def __init__(self):
        self.nirs_obj = new_NIRScanner()
        atexit.register(self._cleanup)

    def _cleanup(self):
        print("Cleanning up NIRS instance.")
        delete_NIRScanner(self.nirs_obj)

    def scan_snr(self, scan_type="hadamard"):
        if scan_type == "hadamard":
            hadamard_flag = True
        elif scan_type == "column":
            hadamard_flag = False
        else:
            print("Unknow scan type {}.".format(scan_type))
        results_str = NIRScanner_scanSNR(self.nirs_obj, hadamard_flag)

        # Convert to Python object and return. 
        return eval(results_str)

    def scan(self, num_repeats=1):
        NIRScanner_scan(self.nirs_obj, False, num_repeats)

    @staticmethod
    def _safe_int(val, default=0):
        """Convert string to int, returning default on empty or invalid input."""
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_float(val, default=0.0):
        """Convert string to float, returning default on empty or invalid input."""
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def get_scan_results(self):
        results_dict = {}
        results_str = NIRScanner_getScanData(self.nirs_obj)
        results_str = results_str.split("\n")

        # Deserialization.
        for item in results_str:
            keyvalue = item.split(":")
            if len(keyvalue) == 2:
                results_dict[keyvalue[0]] = keyvalue[1]

        # Type conversion - all guarded against empty/malformed C++ output.
        if "valid_length" in results_dict:
            length = self._safe_int(results_dict.get("valid_length", "0"))
            results_dict["valid_length"] = length

            # Convert numerical results (filter empty strings from split).
            if "wavelength" in results_dict:
                results_dict["wavelength"] = [
                    self._safe_float(item) for item in results_dict["wavelength"].split(",")[:length]
                    if item.strip()
                ]
            if "intensity" in results_dict:
                results_dict["intensity"] = [
                    self._safe_int(item) for item in results_dict["intensity"].split(",")[:length]
                    if item.strip()
                ]
            if "reference" in results_dict:
                results_dict["reference"] = [
                    self._safe_int(item) for item in results_dict["reference"].split(",")[:length]
                    if item.strip()
                ]

        if "temperature_system" in results_dict:
            results_dict["temperature_system"] = self._safe_int(results_dict["temperature_system"]) / 100.0
        if "temperature_detector" in results_dict:
            results_dict["temperature_detector"] = self._safe_int(results_dict["temperature_detector"]) / 100.0
        if "humidity" in results_dict:
            results_dict["humidity"] = self._safe_int(results_dict["humidity"]) / 100.0
        if "pga" in results_dict:
            results_dict["pga"] = self._safe_int(results_dict["pga"])

        return results_dict

    def scan_collect(self, num_repeats=1, max_retries=1):
        """Scan and collect results with data validation.

        Returns (success: bool, results: dict).
        On failure, clears error status and retries up to max_retries times.
        """
        for attempt in range(max_retries + 1):
            self.scan(num_repeats)
            results = self.get_scan_results()

            intensities = results.get("intensity", [])
            wavelengths = results.get("wavelength", [])
            header = results.get("header_version", "0")

            # Validate: must have real data
            if intensities and wavelengths and len(intensities) > 0:
                # Check for garbage header version (valid headers < 0x100000)
                hv = self._safe_int(header, 0)
                if hv > 0x100000:
                    print(f"[NIRS] WARNING: Suspicious header version {hv} (0x{hv:X}), "
                          f"attempt {attempt+1}/{max_retries+1}")
                    if attempt < max_retries:
                        self.clear_error_status()
                        continue
                    # Last attempt - accept data despite bad header

                # Check for all-zero intensities: indicates _interpretData()
                # failed at the C++ level and mScanResults was zeroed (or
                # retained stale zeros). A real spectrum always has non-zero
                # intensity values.
                if all(v == 0 for v in intensities):
                    print(f"[NIRS] WARNING: All intensities are zero - "
                          f"interpretation likely failed "
                          f"(attempt {attempt+1}/{max_retries+1})")
                    if attempt < max_retries:
                        self.clear_error_status()
                        continue

                return True, results

            # No valid data
            print(f"[NIRS] WARNING: Scan returned no valid data "
                  f"(attempt {attempt+1}/{max_retries+1})")
            if attempt < max_retries:
                self.clear_error_status()

        return False, results

    def display_version(self):
        return NIRScanner_readVersion(self.nirs_obj)

    def set_hibernate(self, new_value: bool):
        return NIRScanner_setHibernate(self.nirs_obj, new_value)

    def set_config(self, scanConfigIndex=8, scan_type=1, num_patterns=228, num_repeats=6, 
                   wavelength_start_nm=900, wavelength_end_nm=1700, width_px=7):
        return NIRScanner_setConfig(self.nirs_obj, scanConfigIndex, scan_type, num_patterns, num_repeats, 
                                    wavelength_start_nm, wavelength_end_nm, width_px)
    
    def set_pga_gain(self, new_value):
        return NIRScanner_setPGAGain(self.nirs_obj, new_value)

    def set_lamp_on_off(self, new_value):
        return NIRScanner_setLampOnOff(self.nirs_obj, new_value)
    
    def clear_error_status(self):
        return NIRScanner_resetErrorStatus(self.nirs_obj)

    def close(self):
        """Explicitly release the C++ device object, calling the destructor
        (which closes the USB HID handle via USB_Close). De-registers from
        atexit to prevent double-free on process exit."""
        if getattr(self, 'nirs_obj', None) is not None:
            try:
                delete_NIRScanner(self.nirs_obj)
            except Exception:
                pass
            self.nirs_obj = None
        try:
            atexit.unregister(self._cleanup)
        except Exception:
            pass


if __name__ == "__main__":
    import time

    nirs = NIRS()
    nirs.display_version()

    # Set config. 
    nirs.set_config(8, NIRS.TYPES.HADAMARD_TYPE, 228, 6, 900, 1700, 7)

    # Turn on the lamp.
    nirs.set_lamp_on_off(True)
    time.sleep(3)

    # Scan.
    nirs.scan()
    results = nirs.get_scan_results()
    nirs.scan()
    results = nirs.get_scan_results()

    # Turn lamp off.
    nirs.set_lamp_on_off(False)

    pass