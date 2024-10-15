import uuid

def get_unique_system_id():
    interface = 'wlp0s20f3' 
    try:
        mac = ni.ifaddresses(interface)[ni.AF_LINK][0]['addr']
        return mac
    except KeyError:
        return f"No MAC address found for {interface}."
    except ValueError:
        return "Invalid interface name."
    # # Get the MAC address as a unique identifier for the system
    # mac_address = uuid.getnode()
    
    # # Format the MAC address into a readable string
    # mac_str = ':'.join(("%012X" % mac_address)[i:i+2] for i in range(0, 12, 2))
    
    return mac_str

# Example usage
unique_id = get_unique_system_id()
print(f"The unique system ID is: {unique_id}")
