# README

## Running the Program

To run this program, you must first create a YAML configuration file containing the required simulation parameters.

### Step 1: Create a Configuration File

* Create a `.yaml` file with your desired configuration settings.
* It is recommended to place this file inside the `configuration_files` directory for better organization.

### Step 2: Set the Configuration Path

* Open the `main.py` script.
* Locate the `load_config` variable where the configuration file path is defined.
* Update this path so that it points to your YAML file.

### Step 3: Run the Program

* Execute the main script:

```bash
python main.py
```

The program will load the configuration from the specified YAML file and start execution.

---

## Notes

* Ensure that the YAML file is correctly formatted to avoid runtime errors (look at the provided example).
* You may create multiple configuration files for different experiments and switch between them by updating the path in `main.py`.
* It is good practice to create one configuration file per experiment for better record-keeping.
