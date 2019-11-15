
DATA_27MARCH2019 = "/home/sandboxes/jbrandt/Telescope27Mar2019"
SCAN9 = "Scan-9_5100x5028_20190327_1145_ReIoNo_ReMxNo_ColorNo_.ptx"
SCAN11 = "Scan-11_5100x5028_20190327_1155_ReIoNo_ReMxNo_ColorNo_.ptx"
SETTINGS_27MARCH2019 = {
    'xOffset': -8.0,
    'yOffset': 50.0,
    'rot': -10.0,
    'radius': 45.5,
    'dataPath': DATA_27MARCH2019
    }
"""
These are the settings used to extract the scans during development.
This does not mean these are optimal parameters.
For example, using a radius smaller than 49 results in a bias in the 
recovered Zernike values during the October scans.
"""
SETTINGS_12JUNE2019 = {
    'xOffset': -8.0,
    'yOffset': 50.0,
    'rot': -10.0,
    'radius': 47.0,
    'dataPath': '/home/scratch/pmargani/LASSI/scannerData/'
    }
SETTINGS_17SEPTEMBER2019 = {
    'xOffset': -45.0,
    'yOffset': -5.0,
    'rot': 80.0,
    'radius': 45.5,
    'dataPath': '/home/sandboxes/pmargani/LASSI/data/17sep2019'
    }
SETTINGS_10OCTOBER2019 = {
    'xOffset': -52.0,
    'yOffset': -8.0,
    'rot': 80.0,
    'radius': 49.0,
    'dataPath': '/home/sandboxes/pmargani/LASSI/data/9oct2019'

    }
SETTINGS_11OCTOBER2019 = {
    'xOffset': -44.0,
    'yOffset': -6.5,
    'rot': 80.0,
    'radius': 49.0,
    'dataPath': '/home/sandboxes/pmargani/LASSI/data/9oct2019'
    }

DATA_UNIT_TESTS = "/home/scratch/pmargani/LASSI/unitTestData/"
