from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, conlist, validator
from pydantic.fields import ModelField
from bipo_fastapi.common import generate_default_daily_date, generate_default_lag_date
from kedro.config import ConfigLoader
from bipo.utils import get_project_path

# initiate to inference.yml
project_path = get_project_path()
conf_loader = ConfigLoader(conf_source=project_path / "conf")
conf_inference_fields = conf_loader.get("inference*")["default_request_fields"]
conf_inference_thresholds = conf_loader.get("inference*")["request_thresholds"]

# define default number of inputs here for OutletAttribute
default_num_inference_date = conf_inference_fields["default_num_inference_date"]
# define default number of inputs here for LagsAttribute
default_num_lag_sales_date = conf_inference_fields["default_num_lag_sales_date"]


class OutletAttribute(BaseModel):
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
        is_pandemic_restrictions (bool): Indicator if the day has any pandemic restrictions.
    """

    # all these inputs will become a dict
    inference_date: List[str] = Field(
        default=generate_default_daily_date(
            conf_inference_fields["default_num_inference_date"]
        )
    )
    cost_centre_code: int = Field(
        default=conf_inference_fields["default_cost_centre_code"]
    )
    location: str = Field(default=conf_inference_fields["default_location"])
    type: str = Field(default=conf_inference_fields["default_type"])
    propensity_factor: List[float] = Field(
        default=[conf_inference_fields["default_propensity_factor"]]
        * default_num_inference_date
    )
    is_raining: List[bool] = Field(
        default=[conf_inference_fields["default_is_raining"]]
        * default_num_inference_date
    )
    temp_max: List[float] = Field(
        default=[conf_inference_fields["default_temp_max"]] * default_num_inference_date
    )
    is_raining: List[bool] = Field(
        default=[conf_inference_fields["default_is_raining"]]
        * default_num_inference_date
    )
    is_public_holiday: List[bool] = Field(
        default=[conf_inference_fields["default_is_public_holiday"]]
        * default_num_inference_date
    )
    is_school_holiday: List[bool] = Field(
        default=[conf_inference_fields["default_is_school_holiday"]]
        * default_num_inference_date
    )
    is_pandemic_restrictions: List[bool] = Field(
        default=[conf_inference_fields["default_is_pandemic_restrictions"]]
        * default_num_inference_date
    )

    # validator for datetime
    @validator("inference_date", pre=True)
    def ensure_datetime(cls, date_list: list):
        """function to check if the date is in YYYY-MM-DD format

        Args:
            date_list (list): list of dates

        Raises:
            ValueError: if date is not in the correct format

        Returns:
            date_list: original date_list is no error is raise
        """
        try:
            # check if all the input date are in the correct datetime format
            [datetime.strptime(value, "%Y-%m-%d").date() for value in date_list]
            return date_list
        except ValueError:
            # If parsing fails, it's not in the correct format
            raise ValueError(f"Input date is not in %Y-%m-%d format")


class LagSalesAttribute(BaseModel):
    """
    A Pydantic model for encapsulating all sales attributes linked to a particular date.

    Each instance of this class represents a comprehensive set of features
    that include weather conditions, marketing data, past sales, and temporal data. These
    features can be used for predicting sales and other retail analytics tasks.

    Attributes:
        date (str): The date associated with lag sales attributes, in "YYYY-MM-DD" format.
        lag_sales (List[float]): A list of total sales from the past 14 days. The first element in the list represents sales from 1 day ago, the second element represents sales from 2 days ago, and so on, up to 14 days ago.
    """

    lag_sales_date: conlist(
        str, min_items=conf_inference_thresholds["min_number_of_lag_days"]
    ) = Field(
        default=generate_default_lag_date(
            conf_inference_fields["default_num_lag_sales_date"]
        )
    )
    lag_sales: conlist(
        float, min_items=conf_inference_thresholds["min_number_of_lag_days"]
    ) = Field(
        default=[conf_inference_fields["default_lag_sales"]]
        * default_num_lag_sales_date
    )

    # validator for datetime
    @validator("lag_sales_date", pre=True)
    def ensure_datetime(cls, date_list: list):
        """function to check if the date is in YYYY-MM-DD format

        Args:
            date_list (list): list of dates

        Raises:
            ValueError: if date is not in the correct format

        Returns:
            date_list: original date_list is no error is raise
        """
        try:
            # check if all the input date are in the correct datetime format
            [datetime.strptime(value, "%Y-%m-%d").date() for value in date_list]
            return date_list
        except ValueError:
            # If parsing fails, it's not in the correct format
            raise ValueError(f"Input date is not in %Y-%m-%d format")


class MarketingAttribute(BaseModel):
    """
    A Pydantic model for encapsulating all marketing attributes linked to a particular date.
    Class only accept 1 camapaign at a time.

    Each instance of this class represents a comprehensive set of features
    that include weather conditions, marketing data, past sales, and temporal data. These
    features can be used for predicting sales and other retail analytics tasks.

    Attributes:
        campaign_name (str): Name of the marketing campaign. (Optional. If nothing provided, it will be None)
        campaign_start_date (str): The start date of a marketing campaign, in "YYYY-MM-DD" format. (Optional. If nothing provided, it will be None)
        campaign_end_date (str): The end date of a marketing campaign, in "YYYY-MM-DD" format. (Optional. If nothing provided, it will be None)
        marketing_channels (List[str]): Individual marketing channels. Default input is from default_marketing_channels
        campaign_total_costs (List[float]): The total cost of each marketing channels from 1 campaign. (Optional. If nothing provided, it will be None)
    """

    campaign_name: Optional[str] = Field(
        default=conf_inference_fields["default_campaign_name"]
    )
    campaign_start_date: Optional[str] = Field(
        default=conf_inference_fields["default_campaign_start_date"]
    )
    campaign_end_date: Optional[str] = Field(
        default=conf_inference_fields["default_campaign_end_date"]
    )
    marketing_channels: Optional[
        conlist(str, min_items=len(conf_inference_fields["default_marketing_channels"]))
    ] = Field(default=conf_inference_fields["default_marketing_channels"])
    total_cost: Optional[
        conlist(
            float, min_items=len(conf_inference_fields["default_marketing_channels"])
        )
    ] = Field(
        default=[conf_inference_fields["default_marketing_total_cost"]]
        * len(conf_inference_fields["default_marketing_channels"])
    )

    # validator for datetime
    @validator("campaign_start_date", "campaign_end_date", pre=True)
    def ensure_datetime(cls, date_string):
        """function to check if the date is in YYYY-MM-DD format

        Args:
            date_string (str): str of start and end date

        Raises:
            ValueError: if date is not in the correct format

        Returns:
            date_list: original date_list is no error is raise
        """
        try:
            # check if all the input date are in the correct datetime format
            datetime.strptime(date_string, "%Y-%m-%d").date()
            return date_string
        except ValueError:
            # If parsing fails, it's not in the correct format
            raise ValueError(f"Input date is not in %Y-%m-%d format")


class SalesAttributes(BaseModel):
    """
    A Pydantic model representing a list of sales attributes.

    Attributes:
        outlet_attributes (List[OutletAttribute]): A list of OutletAttribute objects.
        lag_sales_attributes (List[LagSalesAttribute]): A list of LagSalesAttribute objects.
        mkt_attributes (List[MarketingAttribute]): A list of MarketingAttribute objects.
    """

    # reveune attributes
    outlet_attributes: List[OutletAttribute]

    # lag sales attributes
    lag_sales_attributes: List[LagSalesAttribute]

    # marketing attributes
    mkt_attributes: List[MarketingAttribute]


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
