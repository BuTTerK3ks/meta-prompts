import os

def list_files(directory, max_files=200):
    files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            files_list.append(os.path.join(root, file))
            if len(files_list) >= max_files:
                break
        if len(files_list) >= max_files:
            break

    return files_list

# Set the directory to check
directory_to_check = '/home/grannemann/Allgemein/Christian/LOOXIS/wecreateyour/images'

# List up to 200 files
files = list_files(directory_to_check)

# Print the result
for file in files:
    print(file)

# Print the count of files found
print(f"Total files listed: {len(files)}")