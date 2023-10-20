# Model Serving Endpoint

This document provides detailed information on how to interact with our model serving endpoint, which facilitates predictions through a RESTful API. Ensure you've followed the [`Environment Setup`](inference-deployment-env-setup) before diving in.

## Overview

Our model serving endpoint is designed to take in input data, feed it into a trained model, and deliver the model predictions in real-time. If you're unfamiliar with REST APIs, consider browsing [MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web/API) or [REST API Tutorial](https://restfulapi.net/) for foundational knowledge.

## How to Run Endpoint

### 1. Starting the Endpoint
To run the Docker image containing the endpoint, execute the following command in your home directory. This assumes that your model, data, and configurations are stored in specific directories on your host machine. Note that all path used must be absolute paths.

```bash
$ bash ~/bipo_demand_forecasting/scripts/docker_run.sh
```

### 2. Interacting with the API through Swagger UI

   After starting the Docker container, you can easily interact with the API endpoints using the Swagger UI, a powerful tool for visualizing and testing API endpoints.

1. **Access the Interface**: Open your preferred web browser and navigate to the Swagger UI, typically located at http://<VM_IP_address>:8000/docs
2. **Submit Requests**: The interface provides an interactive platform to send requests and view responses. To initiate a request:
    1. Locate the POST `/api/v1/model/predict` endpoint.
    2. Click the `Try it out` button.
    3. Enter the JSON data conforming to the [Request Format](#request-format).
    4. Click `Execute` to send the request.
3. **Viewing Responses**: After submission, the prediction response will be displayed directly within the Swagger UI. To understand the structure and meaning of the response, refer to the [Response Format](#response-format) section.

      ![Swagger UI Interactive Interface](assets/swaggerUI.png)

### 3. Terminating the Endpoint

- **Foreground Execution**: If the Docker container was started in the foreground, simply press `Ctrl + C` in the terminal to halt it.
- **Background Execution**: If the container is running in the background, use the following command to stop it:

```bash
docker stop --name bipo_inference_initial
```
Note: Replace `bipo_inference:initial` with the name you used when starting the container, if different.

## Request Format

To obtain predictions, format your POST request as follows: 

```json
{
    "sales_attributes": [
        {
            "date": "20/3/2023",
            "cost_centre_code": 308,
            "location": "East",
            "type": "dine-in",
            "propensity_factor": 1.5,
            "is_raining": "FALSE",
            "max_temp": 25.8,
            "is_public_holiday": "TRUE",
            "is_school_holiday": "TRUE",
            "campaign_name": "pizza_hut_promo1",
            "campaign_start_date": "1/3/2023",
            "campaign_end_date": "26/3/2023",
            "campaign_total_costs": 20100,
            "lag_sales": [
                20100,
                20200,
                20300,
                20400,
                20500,
                20600,
                20700,
                20800,
                20900,
                21000,
                21100,
                21200,
                21300,
                21400
            ]
        },
        {
            "date": "21/3/2023",
            "cost_centre_code": 308,
            "location": "East",
            "type": "dine-in",
            "propensity_factor": 1.5,
            "is_raining": "TRUE",
            "max_temp": 25.8,
            "is_public_holiday": "TRUE",
            "is_school_holiday": "TRUE",
            "campaign_name": "pizza_hut_promo1",
            "campaign_start_date": "1/3/2023",
            "campaign_end_date": "26/3/2023",
            "campaign_total_costs": 20100,
            "lag_sales": [
                20100,
                20200,
                20300,
                20400,
                20500,
                20600,
                20700,
                20800,
                20900,
                21000,
                21100,
                21200,
                21300,
                21400
            ]
        }
    ]
}
```
| Attribute             | Type        | Description                                                                                                   |
| --------------------- | ----------- | ------------------------------------------------------------------------------------------------------------- |
| `sales_attributes`    | list        | List of data points used for predicting sales.                                                                |
| `date`                | str         | Date linked to the sales attributes, formatted as "DD/MM/YYYY".                                               |
| `cost_centre_code`    | int         | Numeric code identifying the cost centre.                                                                     |
| `location`            | str         | Geographical point where the sale occurred.                                                                   |
| `type`                | str         | Category or classification of the sale (e.g., retail, online).                                                |
| `propensity_factor`   | float       | Score indicating likelihood or propensity of sales, with higher values signaling greater likelihood.          |
| `is_raining`          | bool        | True if it rained on the given date; otherwise False.                                                         |
| `max_temp`            | float       | Day's highest recorded temperature in degrees.                                                                |
| `is_public_holiday`   | bool        | True if the date is a public holiday; otherwise False.                                                        |
| `is_school_holiday`   | bool        | True if the date falls within school holiday periods; otherwise False.                                        |
| `campaign_name`       | str         | Designation of the associated marketing campaign.                                                             |
| `campaign_start_date` | str         | Date when the marketing campaign commenced, formatted as "DD/MM/YYYY".                                        |
| `campaign_end_date`   | str         | Date when the marketing campaign concluded, formatted as "DD/MM/YYYY".                                        |
| `campaign_total_cost` | float       | Aggregate expenditure associated with the marketing campaign.                                                 |
| `lag_sales`           | List[float] | Sequential list of sales figures from the last 14 days, starting from the previous day up to two weeks prior. |

### Optional Fields
If not applicable (i.e. no ongoing marketing campaign during the inference period), remove the following 4 fields completely from the request.
- `campaign_name`
- `campaign_start_date`
- `campaign_end_date`
- `campaign_total_cost`

For `lag_sales`, leave the field as `[ ]` if not applicable.

## Response Format

The server's response contains:

```json
{
    "sales_predictions": [
        {
            "date": "20/3/2023",
            "cost_centre_code": 308,
            "sales_class_id": "3",
            "sales_class_name": "Exceptional",
            "probabilities": {
                "0": 0.0,
                "1": 0.1429,
                "2": 0.2857,
                "3": 0.5714
            }
        },
        {
            "date": "21/3/2023",
            "cost_centre_code": 308,
            "sales_class_id": "2",
            "sales_class_name": "High",
            "probabilities": {
                "0": 0.1611,
                "1": 0.2280,
                "2": 0.3284,
                "3": 0.2826
            }
        }
    ]
}
```
| Attribute           | Type             | Description                                                                                             |
| ------------------- | ---------------- | ------------------------------------------------------------------------------------------------------- |
| `sales_predictions` | list             | List of sales forecasts corresponding to input data.                                                    |
| `date`              | str              | Date of the prediction in "DD/MM/YYYY" format, matching the input API request.                          |
| `cost_centre_code`  | int              | Identifier for a specific outlet, matching the input API request.                                       |
| `sales_class_id`    | str              | ID linked to the class with the highest probability from the `probabilities` attribute.                 |
| `sales_class_name`  | str              | Descriptive name based on `sales_class_id` and the highest probability.                                 |
| `probabilities`     | Dict[str, float] | Dictionary mapping class IDs to likelihood scores, determining `sales_class_id` and `sales_class_name`. |

### Mapping of `sales_class_id` to `sales_class_name`: 

| `sales_class_id` |       `sales_class_name`       |
|---|---|
| 0 |       Low       |
| 1 |       Medium       |
| 2 |       High       |
| 3 |       Exceptional       |

The `date` and `cost_centre_code` attributes serve as a mapping to the respective fields in the input API request payload, ensuring alignment and traceability between the request and response.

#### Example

For the entry on "20/3/2023" with a "cost_centre_code" of 308, the probabilities are {"0": 0.0, "1": 0.1429, "2": 0.2857, "3": 0.5714}. 
- As 0.5714 is the highest probability, the corresponding `sales_class_id` is "3" and `sales_class_name` is "Exceptional". 
  
Similarly, for "21/3/2023", the highest probability is 0.3284, resulting in a `sales_class_id` of "2" and a `sales_class_name` of "High".

## API Error Codes

When interacting with the API, you may occasionally encounter error responses. To help you navigate these situations, we've provided explanations for some of the more common error status codes:

|Error code|Description|
|---|---|
|400 (Bad Request)|This indicates that the server couldn't understand the request due to invalid syntax or format. Before making another request:<ul><li>Ensure that the API request payload adheres to the expected format.</li><li>Double-check against the [Request Format](#request-format) section to ensure accuracy.</li></ul>|
|500 (Internal Server Error)|This is a catch-all error, indicating unexpected conditions on the server side. Possible steps to troubleshoot include:<ul><li>Review server logs for detailed error messages.</li><li> Ensure the server environment and dependencies are correctly set up.</li><li> If the issue persists, reach out to the technical support or the system administrator for assistance.|
  
For a deeper dive into HTTP status codes and their meanings, you can explore the [HTTP Status Codes on RESTful API](https://restfulapi.net/http-status-codes/).
