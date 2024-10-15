import requests
import paramiko
import time

def get_public_ip():
    try:
        # Using an external service to fetch public IP
        response = requests.get('https://api.ipify.org?format=json')
        data = response.json()
        return data['ip']
    except requests.RequestException as e:
        return f"Error fetching IP: {e}"

def allow_mongo_access_from_ip(ip_address):
    # Define the SSH connection details
    hostname = "202.83.16.6"
    username = "lset-01"
    password = "Lset@1234"
    sudo_password = "Lset@1234"  # Assuming the sudo password is the same

    # Create an SSH client
    client = paramiko.SSHClient()

    # Automatically add the server's SSH key (without user confirmation)
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect to the server
        client.connect(hostname, username=username, password=password)

        # Command to allow the public IP on MongoDB port 27017 using UFW
        command = f"sudo ufw allow from {ip_address} to any port 27017"
        stdin, stdout, stderr = client.exec_command(command, get_pty=True)

        # Provide the sudo password when prompted
        stdin.write(f"{sudo_password}\n")
        stdin.flush()

        # Wait for the command to complete
        time.sleep(2)

        # Get the output and errors
        output = stdout.read().decode()
        error = stderr.read().decode()

        # Print the output and errors
        if output:
            print(f"Output: {output}")
        if error:
            print(f"Error: {error}")

        # Close the connection
        client.close()

    except Exception as e:
        print(f"Failed to connect or execute command: {str(e)}")

# # Get the public IP address
# public_ip = get_public_ip()

# if 'Error' not in public_ip:
#     print(f"My Public IP Address is: {public_ip}")
#     # Allow access to MongoDB from this IP
#     allow_mongo_access_from_ip(public_ip)
# else:
#     print(f"Failed to fetch public IP: {public_ip}")

