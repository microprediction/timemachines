import requests

CHECKED_INTERNET=False
CONNECTED_TO_INTERNET=None


def connected_to_internet():
    global CHECKED_INTERNET
    global CONNECTED_TO_INTERNET
    if not CHECKED_INTERNET:
        CONNECTED_TO_INTERNET = check_if_connected_to_internet()
    return CONNECTED_TO_INTERNET


def check_if_connected_to_internet(url='http://www.google.com/', timeout=2):
    try:
        _ = requests.head(url, timeout=timeout)
        return True
    except requests.ConnectionError:
        print("No internet connection available.")
    return False
