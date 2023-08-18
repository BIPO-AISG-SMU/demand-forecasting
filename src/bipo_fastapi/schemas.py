from typing import List, Dict
from pydantic import BaseModel, Field


class SalesAttribute(BaseModel):
    """
    A Pydantic model for encapsulating all sales attributes linked to a particular date.

    Each instance of this class represents a comprehensive set of features
    that include weather conditions, marketing data, past sales, and temporal data. These
    features can be used for predicting sales and other retail analytics tasks.

    Attributes:
        date (str): The date associated with these attributes, in "YYYY-MM-DD" format.
        cost_centre_code (int): The code of the cost centre.
        location (str): The geographical location of the sale.
        type (str): The type of sale.
        propensity_factor (float): A measure indicating the propensity of sales.
        is_raining (bool): Indicator variable representing whether it was raining on the day.
        max_temp (float): Maximum temperature for the day.
        is_public_holiday (bool): Indicator if the day was a public holiday.
        is_school_holiday (bool): Indicator if the day was during school holidays.
        campaign_name (str): Name of the marketing campaign.
        campaign_start_date (str): The start date of a marketing campaign, in "YYYY-MM-DD" format.
        campaign_end_date (str): The end date of a marketing campaign, in "YYYY-MM-DD" format.
        campaign_total_costs (float): The total cost of the marketing campaign.
        lag_sales (List[float]): A list of total sales from the past 14 days. The first element in the list represents sales from 1 day ago, the second element represents sales from 2 days ago, and so on, up to 14 days ago.
    """

    date: str
    cost_centre_code: int
    location: str
    type: str
    propensity_factor: float
    is_raining: bool
    max_temp: float
    is_public_holiday: bool
    is_school_holiday: bool
    campaign_name: str | None = Field(default=None)
    campaign_start_date: str | None = Field(default=None)
    campaign_end_date: str | None = Field(default=None)
    campaign_total_costs: float | None = Field(default=None)
    lag_sales: List[float] | None = Field(default=[])


class SalesAttributes(BaseModel):
    """
    A Pydantic model representing a list of sales attributes.

    Attributes:
        sales_attributes (List[SalesAttribute]): A list of SalesAttribute objects.
    """

    sales_attributes: List[SalesAttribute]


class SalesPrediction(BaseModel):
    """
    A Pydantic model representing a single sales prediction.

    Attributes:
        date (str): The date associated with this prediction.
        cost_centre_code (int): The code of the cost centre.
        sales_class_id (str): The class ID associated with this prediction.
        sales_class_name (str): The class name associated with this prediction.
        probabilities (Dict[str, float]): A dictionary of probabilities associated with this prediction.
    """

    date: str
    cost_centre_code: int
    sales_class_id: str
    sales_class_name: str
    probabilities: Dict[str, float]


class SalesPredictions(BaseModel):
    """
    A Pydantic model representing a list of sales predictions.

    Attributes:
        sales_predictions (List[SalesPrediction]): A list of SalesPrediction objects.
    """

    sales_predictions: List[SalesPrediction]
