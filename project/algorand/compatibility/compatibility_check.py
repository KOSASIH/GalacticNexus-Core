# project/algorand/compatibility/compatibility_check.py

import pkg_resources

def check_algorand_sdk_version(required_version):
    """
    Check if the installed Algorand SDK version meets the required version.
    """
    try:
        algorand_sdk_version = pkg_resources.get_distribution("algosdk").version
        if algorand_sdk_version < required_version:
            print(f"Warning: Installed Algorand SDK version {algorand_sdk_version} is less than the required version {required_version}.")
        else:
            print(f"Algorand SDK version {algorand_sdk_version} is compatible.")
    except pkg_resources.DistributionNotFound:
        print("Error: Algorand SDK is not installed.")

if __name__ == "__main__":
    required_version = "1.10.0"  # Example required version
    check_algorand_sdk_version(required_version)
