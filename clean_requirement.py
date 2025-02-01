import pkg_resources  # Use to get installed package versions

def fetch_package_version(package_name):
    """
    Fetch the installed version of a package.
    :param package_name: Name of the package to query.
    :return: Installed version of the package or "UNKNOWN" if not found.
    """
    try:
        # Get the package version from installed distributions
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return "UNKNOWN"
    except Exception as e:
        print(f"Error fetching version for {package_name}: {e}")
        return "UNKNOWN"

def clean_and_fix_requirements(file_path):
    """
    Clean the requirements file and add missing version numbers.
    :param file_path: Path to the requirements.txt file.
    """
    # Read the file content
    with open(file_path, "r") as infile:
        lines = infile.readlines()

    cleaned_lines = []
    for line in lines:
        # Clean lines with "@ file://"
        if "@ file://" in line:
            package_version = line.split("@")[0].strip()
            if "==" in package_version:
                package, version = package_version.split("==")
                cleaned_lines.append(f"{package}=={version.strip()}\n")
            else:
                cleaned_lines.append(f"{package_version}\n")
        # Add regular lines with "=="
        elif "==" in line:
            cleaned_lines.append(line)
        # Handle lines without versions (fetch versions dynamically)
        else:
            package_name = line.strip()
            if package_name:  # Ignore empty lines
                version = fetch_package_version(package_name)
                if version != "UNKNOWN":
                    cleaned_lines.append(f"{package_name}=={version}\n")
                else:
                    print(f"Warning: Version not found for package '{package_name}'")

    # Write the cleaned and fixed content back to the file
    with open(file_path, "w") as outfile:
        outfile.writelines(cleaned_lines)

    print("Requirements file cleaned and fixed successfully!")

# Run the function
clean_and_fix_requirements("requirements.txt")
