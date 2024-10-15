from pymongo import MongoClient
import uuid
import random
import netifaces as ni
# Function to get MAC address
def get_mac_address():
    interface = 'wlp0s20f3' 
    try:
        mac = ni.ifaddresses(interface)[ni.AF_LINK][0]['addr']
        return mac
    except KeyError:
        return f"No MAC address found for {interface}."
    except ValueError:
        return "Invalid interface name."

# Function to check MAC address and return only _id
def check_and_insert_mac_address(client_uri, db_name, collection_name):
    # MongoDB client setup
    client = MongoClient(client_uri)

    # Access the specified collection in the specified database
    db = client[db_name]
    source_collection = db[collection_name]

    # Get MAC address of the current system
    mac_address = get_mac_address()
    print(mac_address)
    print("&&***^^^^^^%%%%%%%%%%"*100)

    # Search for the MAC address in the collection
    existing_record = source_collection.find_one({"mac address": mac_address})

    if existing_record:
        # If MAC address exists, return the _id
        print(f"MAC address already exists in the {collection_name} collection.")
        return existing_record.get("_id")
    else:
        # If MAC address is not found, get any random existing document
        reference_record = source_collection.find_one()

        if reference_record:
            # Create a new record based on the reference with a new _id and the MAC address
            new_record = reference_record.copy()  # Copy the reference record
            new_record["_id"] = random.randint(1000, 9999)  # Generate a random _id
            new_record["mac address"] = mac_address  # Add the new MAC address

            # Insert the new record into the collection
            source_collection.insert_one(new_record)
            print(f"New record inserted in the {collection_name} collection.")
            return new_record.get("_id")
        else:
            print(f"No reference record found in the {collection_name} collection to copy.")
            return None

# # Example usage
# client_uri = 'mongodb://LSET:LSET432@202.83.16.6:27017/'
# db_name = 'analytics'
# collection_name = 'source'

# record_id = check_and_insert_mac_address(client_uri, db_name, collection_name)
# print(f"_id: {record_id}")
