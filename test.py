# Read binary content of the pickled file
with open('data.pkl', 'rb') as f:
    binary_data = f.read()

# Display the binary content in hexadecimal format
hexadecimal_representation = ' '.join(f'{byte:02X}' for byte in binary_data)
print("Hexadecimal Representation of Pickled Data:")
print(hexadecimal_representation)
