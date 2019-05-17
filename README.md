# senzing-mapping-assistant

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

## Tests

1. Using the following `bash` commands:

    ```console
    export CLASSIFICATION_FILES=( \
      "addr_city" \
      "addr_state" \
      "drivers_license_number" \
      "entityid" \
      "geo_latlong" \
      "name_last" \
      "phone_number" \
      "srccode" \
      "addr_line1" \
      "cc_account_number" \
      "drivers_license_state" \
      "entity_type" \
      "hbase_row_key" \
      "name_suffix" \
      "record_id" \
      "ssn_number" \
      "addr_postal_code" \
      "date_of_birth" \
      "dsrc_action" \
      "gender" \
      "name_first" \
      "passport_number" \
      "social_handle" \
    )

    for CLASSIFICATION_FILE in ${CLASSIFICATION_FILES[@]}; \
    do \
      export INPUT_FILE="test/${CLASSIFICATION_FILE}/${CLASSIFICATION_FILE^^}.txt"; \
      python ./senzing-mapping-assistant.py suggest-as-markdown --input-file ${INPUT_FILE}
    done
    ```

## Results

### ADDR_CITY.txt

```console
89.8% - addr_city
 7.0% - name_last
 2.2% - name_first
```

### ADDR_STATE.txt

```console
59.9% - addr_state
38.6% - drivers_license_state
 1.2% - addr_city
```

### DRIVERS_LICENSE_NUMBER.txt

```console
93.6% - addr_city
 1.8% - phone_number
```

### GEO_LATLONG.txt

```console
56.8% - geo_latlong
25.0% - addr_city
15.3% - ssn_number
 2.8% - date_of_birth
```

### NAME_LAST.txt

```console
73.1% - name_last
23.3% - addr_city
 2.9% - name_first
```

### PHONE_NUMBER.txt

```console
99.2% - phone_number
```

### ADDR_LINE1.txt

```console
85.1% - addr_line1
 4.3% - phone_number
 3.9% - drivers_license_state
 1.5% - ssn_number
```

### CC_ACCOUNT_NUMBER.txt

```console
99.0% - addr_city
```

### DRIVERS_LICENSE_STATE.txt

```console
85.7% - drivers_license_state
14.3% - addr_state
```

### ENTITY_TYPE.txt

```console
100.0% - entity_type
```

### HBASE_ROW_KEY.txt

```console
99.9% - hbase_row_key
```

### NAME_SUFFIX.txt

```console
80.5% - name_suffix
19.5% - addr_city
```

### RECORD_ID.txt

```console
99.7% - addr_city
```

### SSN_NUMBER.txt

```console
59.2% - ssn_number
21.2% - date_of_birth
19.5% - geo_latlong
```

### ADDR_POSTAL_CODE.txt

```console
85.1% - addr_postal_code
 7.5% - addr_city
 2.7% - phone_number
 1.7% - hbase_row_key
 1.2% - drivers_license_number
 1.1% - addr_line1
```

### DATE_OF_BIRTH.txt

```console
99.9% - date_of_birth
```

### DSRC_ACTION.txt

```console
100.0% - addr_city
```

### GENDER.txt

```console
100.0% - addr_city
```

### NAME_FIRST.txt

```console
77.2% - name_first
17.8% - addr_city
 4.6% - name_last
```

### PASSPORT_NUMBER.txt

```console
99.5% - addr_city
```

### SOCIAL_HANDLE.txt

```console
92.2% - addr_city
 7.5% - social_handle
```

### Errors

1. [DRIVERS_LICENSE_NUMBER.txt](#drivers-license-numbertxt)
1. [CC_ACCOUNT_NUMBER.txt](#cc-account-numbertxt)
1. [RECORD_ID.txt](#record-idtxt)
1. [DSRC_ACTION.txt](#dsrc-actiontxt)
1. [GENDER.txt](#gendertxt)
1. [PASSPORT_NUMBER.txt](#passport-numbertxt)
1. [SOCIAL_HANDLE.txt](#social-handletxt)
