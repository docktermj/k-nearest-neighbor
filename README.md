# k-nearest-neighbor


## The Experiment

1. Download example records.
   Example:

    ```console
    wget https://s3.amazonaws.com/public-read-access/TestDataSets/loadtest-dataset-1M_shuffled.json
    ```

1. Split the file in half into a training dataset and a test dataset.
   Example:

    ```console
    split -l 500000 loadtest-dataset-1M_shuffled.json
    ```

1. Rename the split files to easily identify the training and test datasets.
   Example:

    ```console
    mv xaa loadtest-dataset-training.json
    mv xab loadtest-dataset-test.json
    ```

1. Using the training dataset, read each JSONLINE and aggregate values by the JSON key.
   In otherwords, all values for the "ADDR_CITY" JSON key will be appended to the `ADDR_CITY.txt` file.
   Example:

    ```console
    python ./senzing-mapping-assistant.py prepare --jsonlines-file loadtest-dataset-training.json
    ```

1. Train a model. Given the plethora of files, each containing values for a single JSON key,
   create a classification model to determine the characteristics of what values belong to each JSON key.
   Example:

    ```console
    python ./senzing-mapping-assistant.py train
    ```

1. Dissect the test dataset into multiple files.
   Each file contains the values for a single JSON key from the test dataset.
   The files will be placed in the `test` directory.
   This was done before for the training dataset.
   Example:

    ```console
    python ./senzing-mapping-assistant.py prepare --jsonlines-file loadtest-dataset-test.json --output-directory test
    ```

1. Using the values from a single JSON key,
   see if `senzing-mapping-assistant.py` can determine the most likely classification
   for the set of values in the test dataset.

    ```console
    python ./senzing-mapping-assistant.py suggest --pretty --input-file test/addr_city/ADDR_CITY.txt
    ```

   Results:

    ```console
    89.8% - addr_city
     7.0% - name_last
     2.2% - name_first
    ```
